# from trainer_Synapse.py

import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from loss.TverskyLoss import TverskyLoss_binary
from torchvision.utils import save_image
from scripts.map_generator import overlay, save_color_heatmap
from scripts.inference import inference

def make_worker_init_fn(base_seed: int):
    def _init(worker_id: int):
        seed = base_seed + worker_id
        random.seed(seed); 
        np.random.seed(seed); 
        torch.manual_seed(seed)
    return _init

def trainer(args, model, log_save_path = "", config = None):
    from dataset.dataset import SegArtifact_dataset, RandomGenerator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # folder for the final output of binary and heatmap masks
    os.makedirs(log_save_path, exist_ok=True)
    pred_dir = os.path.join(log_save_path, "final_preds")
    os.makedirs(pred_dir, exist_ok=True)

    # logger config
    logging.basicConfig(
        filename = log_save_path + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', 
        datefmt='%H:%M:%S') #houres, minutes, seconds
    # every log is visible in terminal and log-file
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args)) # logs the args

    if config == None: 
        logging.error("Config file is not found!")
        raise ValueError("Config file is not found!")
 
    freeze_encoder = args.freeze_encoder

    base_lr = args.base_lr                      # learning rate
    num_classes = args.num_classes              # number of classes
    if args.n_gpu > 0:
        batch_size = args.batch_size * args.n_gpu   # batch_size
    else:
        batch_size = args.batch_size

    # preparation of data
    db_train = SegArtifact_dataset( base_dir = args.root_path, 
                                    list_dir = args.list_dir, 
                                    split = "train",
                                    transform = transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    
    logging.info("The length of train set is: {}".format(len(db_train)))

    trainloader = DataLoader(   db_train, 
                                batch_size = batch_size, 
                                shuffle = True,                     # shuffles the data sequence
                                num_workers = 8,                    # number of parallel processes (Number_of_CPU_cores / 2)
                                pin_memory = torch.cuda.is_available(),           # true if GPU available
                                worker_init_fn=make_worker_init_fn(args.seed) )   # worker seed
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    def core(m):  
        return m.module if isinstance(m, nn.DataParallel) else m 
    
    # freez encoder if wanted
    core(model).freeze_encoder(freeze_encoder)
    
    # training modus
    model.train()

    tversky_loss = TverskyLoss_binary(config.TRAIN.TVERSKY_LOSS_ALPHA, config.TRAIN.TVERSKY_LOSS_BETA)

    # AdamW Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr = base_lr,
        betas=(config.TRAIN.OPTIMIZER.MOMENTUM, 0.999),   # "Momentum"-Parameter
        eps = config.TRAIN.OPTIMIZER.EPS,             # kleine Konstante für Stabilität
        weight_decay = config.TRAIN.WEIGHT_DECAY,    # L2-Regularisierung (entkoppelt!)
        amsgrad = False         # optional, selten genutzt
    )
                                                    
    writer = SummaryWriter(log_save_path + '/log')

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    # parameters for unfreezing the encoder:
    stage3_unfreeze = max_epoch*args.unfreeze_stage3
    stage2_unfreeze = max_epoch*args.unfreeze_stage2
    stage1_unfreeze = max_epoch*args.unfreeze_stage1
    stage0_unfreeze = max_epoch*args.unfreeze_stage0
    # creates progress bar
    iterator = tqdm(range(max_epoch), ncols=70,  dynamic_ncols=True)

    bool_s3_unfreezed = False
    bool_s2_unfreezed = False
    bool_s1_unfreezed = False
    bool_s0_unfreezed = False

    best_val_dice = -1.0
    since_best = 0 # counter that counts the number of epochs during which the soft_dice has not improved
    mean_metrics = np.zeros(7, dtype=np.float64)

    for epoch_num in iterator:
        model.train()
    
        # -------- UNFREEZING THE ENCODER --------
        if freeze_encoder:
            # unfreeze form the deepest encoder level to the highests
            if epoch_num > 1 :
                if (epoch_num >= stage3_unfreeze) and (bool_s3_unfreezed == False):
                    core(model).unfreeze_encoder(3)
                    bool_s3_unfreezed = True
                if (epoch_num >= stage2_unfreeze) and (bool_s2_unfreezed == False):
                    core(model).unfreeze_encoder(2)
                    bool_s2_unfreezed = True
                if (epoch_num >= stage1_unfreeze) and (bool_s1_unfreezed == False):
                    core(model).unfreeze_encoder(1)
                    bool_s1_unfreezed = True
                if (epoch_num >= stage0_unfreeze) and (bool_s0_unfreezed == False):
                    core(model).unfreeze_encoder(0)
                    bool_s0_unfreezed = True
        # -------------------------------------------

        # get the batches (image & label) from the DataLoader
        for i_batch, sampled_batch in enumerate(trainloader):
            is_last_epoch = (epoch_num >= max_epoch - 1)
            is_last_batch = (i_batch == len(trainloader) - 1)

            image_batch, label_batch = sampled_batch['image'].to(device), sampled_batch['label'].to(device)
            case_names  = sampled_batch['case_name']

            logging.info("before model \n")
            outputs = model(image_batch)
            logging.info("after model \n")
            
            # loss
            loss = tversky_loss(outputs, label_batch)

            # backprop + optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # poly-learning rate
            # learning rate slowly drops to 0
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

            # In the last epoch, bit masks and heat masks are created for the last batch.
            if is_last_epoch and is_last_batch:
                logging.info("Last epoch, last batch")
                model.eval()
                with torch.no_grad():
                    create_bin_heat_mask(outputs, case_names, pred_dir, image_batch)
                model.train()

        # -------- VALIDATION (aftre every Epoch-Train) --------
        model.eval()
        mean_metrics= inference(model,logging, log_save_path, device, args.root_path, 
            "val", args.list_dir, args.img_size, args.sig_threshold)
        
        mean_dice, mean_IoU, mean_recall, mean_precision, mean_f1_score, mean_soft_dice, mean_soft_IoU = mean_metrics
        write_to_writer(writer, mean_metrics, epoch_num)
        
        if mean_soft_dice > best_val_dice:
            best_val_dice = mean_soft_dice
            since_best = 0
            best_path = os.path.join(log_save_path, 'best_model.pth')
            torch.save({
                'epoch': epoch_num,
                'model': core(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter_num': iter_num,
                'best_val_dice': best_val_dice
            }, best_path)
            logging.info(f"Saved new BEST checkpoint to {best_path} (val_dice={best_val_dice:.5f})")
        else:
            since_best += 1
            if since_best >= args.early_stopping_patience:
                logging.info(f"Early stopping at epoch {epoch_num} (no val improvement for {args.early_stopping_patience} epochs).")
                break
        # --------------------------------------------------------
        
        # saves all 50 epochs
        save_interval = 50 

        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(log_save_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save({    'epoch': epoch_num,
                            'model': core(model).state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'iter_num': iter_num,
                            'soft_dice': mean_soft_dice}
                            , save_mode_path,)
            logging.info("save model to {}".format(save_mode_path))

        # saves the last run
        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(log_save_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save({    'epoch': epoch_num,
                            'model':  core(model).state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'iter_num': iter_num,
                            'soft_dice': mean_soft_dice}, 
                            save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"

def create_bin_heat_mask(outputs, case_names, pred_dir, image_batch):
    logits = outputs                     # (image_pre,1,H,W)
    heat   = torch.sigmoid(logits)       # (B,1,H,W) in [0,1]
    binmsk = (heat > 0.5).float()        # (B,1,H,W) 0/1
    batch_size = heat.shape[0]
    for image_pre in range(batch_size):
        img_name = str(case_names[image_pre])
        # heatmap
        save_image(heat[image_pre], os.path.join(
            pred_dir, f"{img_name}_grey_heats.png"))
        # binmap
        save_image(binmsk[image_pre], os.path.join(
            pred_dir, f"{img_name}_bin_mask.png"))
        
        save_color_heatmap(
            img_3chw = image_batch[image_pre].detach().cpu(),
            heat_hw  = heat[image_pre,0].detach().cpu(), 
            out_png  = os.path.join(pred_dir, f"{img_name}_overlay_color.png"),
            alpha    = 0.45 )       
        # overlay(image_batch, heat, binmsk, pred_dir, image_pre, case_names)

def write_to_writer(writer, mean_metrics, epoch_num):
    mean_dice, mean_IoU, mean_recall, mean_precision, mean_f1_score, mean_soft_dice, mean_soft_IoU = mean_metrics
    writer.add_scalar('val/mean_soft_dice', mean_soft_dice, epoch_num)
    writer.add_scalar('val/mean_soft_IoU',  mean_soft_IoU,  epoch_num)
    writer.add_scalar('val/mean_dice',      mean_dice,      epoch_num)
    writer.add_scalar('val/mean_IoU',       mean_IoU,       epoch_num)
    writer.add_scalar('val/mean_recall',    mean_recall,    epoch_num)
    writer.add_scalar('val/mean_precision', mean_precision, epoch_num)
    writer.add_scalar('val/mean_f1',        mean_f1_score,  epoch_num)
