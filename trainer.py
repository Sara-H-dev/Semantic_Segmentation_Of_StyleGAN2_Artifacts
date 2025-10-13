# from trainer_Synapse.py

import logging
import os
import random
import sys
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from timm.scheduler.cosine_lr import CosineLRScheduler

from tqdm import tqdm
from loss.TverskyLoss import TverskyLoss_binary
from loss.SymmetricUnifiedFocalLoss_2 import SymmetricUnifiedFocalLoss
from loss.DynamicLoss import DynamicLoss
from scripts.map_generator import overlay, save_color_heatmap
from scripts.inference import inference, validation_loss

def make_worker_init_fn(base_seed: int):
    def _init(worker_id: int):
        seed = base_seed + worker_id
        random.seed(seed); 
        np.random.seed(seed); 
        torch.manual_seed(seed)
    return _init

def trainer(model, log_save_path = "", config = None, base_lr = 5e-4):
    from dataset.dataset import SegArtifact_dataset, RandomGenerator

    warmup_epochs = config.TRAIN.WARMUP_EPOCHS
    max_epoch = config.TRAIN.MAX_EPOCHS

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

    if config == None: 
        logging.error("Config file is not found!")
        raise ValueError("Config file is not found!")
 
    freeze_encoder = config.MODEL.FREEZE_ENCODER 
    n_gpu = config.HARDWARE.N_GPU

    img_size = config.DATA.IMG_SIZE
    batch_size = config.DATA.BATCH_SIZE

    if n_gpu > 0:
        batch_size = batch_size * n_gpu   # batch_size


    # preparation of data
    db_train = SegArtifact_dataset( base_dir = config.DATA.DATA_PATH, 
                                    list_dir = config.LIST_DIR, 
                                    split = "train",
                                    transform = transforms.Compose(
                                        [RandomGenerator(output_size=[img_size, img_size], random_flip_flag = True)]))
    
    logging.info("The length of train set is: {}".format(len(db_train)))

    trainloader = DataLoader(   db_train, 
                                batch_size = batch_size, 
                                shuffle = True,                     # shuffles the data sequence
                                num_workers = config.DATA.NUM_WORKERS,                    # number of parallel processes (Number_of_CPU_cores / 2)
                                pin_memory = config.DATA.PIN_MEMORY and torch.cuda.is_available(),           # true if GPU available
                                worker_init_fn=make_worker_init_fn(config.SEED) )   # worker seed
    if n_gpu > 1:
        model = nn.DataParallel(model)

    """calculate log learning rate increase:"""
    min_lr = 1e-6
    max_lr = 1e-2  #1e-5
    num_batches = len(trainloader)
    mult = (max_lr / min_lr) ** (1 / (num_batches * max_epoch))

    def core(m):  
        return m.module if isinstance(m, nn.DataParallel) else m 
    
    # freez encoder if wanted
    core(model).freeze_encoder(freeze_encoder)
    
    # training modus
    model.train()

    # tversky_loss = TverskyLoss_binary(config.TRAIN.TVERSKY_LOSS_ALPHA, config.TRAIN.TVERSKY_LOSS_BETA)
    # uf_loss = SymmetricUnifiedFocalLoss(weight = config.TRAIN.UF_LOSS_WEIGTH, delta = config.TRAIN.UF_LOSS_DELTA, gamma=config.TRAIN.UF_LOSS_GAMMA)
    dynamic_loss = DynamicLoss(alpha=config.TRAIN.TVERSKY_LOSS_ALPHA, beta=config.TRAIN.TVERSKY_LOSS_BETA)

    logging.info(f"Weight_decay = {config.TRAIN.WEIGHT_DECAY}")
    # AdamW Optimizer
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = base_lr,
        betas= config.TRAIN.OPTIMIZER.BETAS,   # "Momentum"-Parameter
        eps = config.TRAIN.OPTIMIZER.EPS,             # kleine Konstante für Stabilität
        weight_decay = config.TRAIN.WEIGHT_DECAY,    # L2-Regularisierung (entkoppelt!)
        amsgrad = False         # optional, selten genutzt
    )

    # Cosine Decay with linear warmup
    """
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial= max_epoch - warmup_epochs,
        t_mul = 1.,
        lr_min = config.TRAIN.MIN_LR,
        warmup_lr_init = config.TRAIN.WARMUP_LR,
        warmup_t = warmup_epochs,
        cycle_limit = 1,
        t_in_epochs=  True,
        warmup_prefix = config.TRAIN.LR_SCHEDULER.WARMUP_PREFIX
    )
    """
                                                    
    writer = SummaryWriter(log_save_path + '/log')

    iter_num = 0
    print(f"lenght trainloader {len(trainloader)}", file=sys.stderr)
    max_iterations = max_epoch * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    
    best_performance = 0.0
    # parameters for unfreezing the encoder:
    stage3_unfreeze = max_epoch * config.MODEL.STAGE3_UNFREEZE_PERIODE
    stage2_unfreeze = max_epoch * config.MODEL.STAGE2_UNFREEZE_PERIODE
    stage1_unfreeze = max_epoch * config.MODEL.STAGE1_UNFREEZE_PERIODE
    stage0_unfreeze = max_epoch * config.MODEL.STAGE0_UNFREEZE_PERIODE

    bool_s3_unfreezed = False
    bool_s2_unfreezed = False
    bool_s1_unfreezed = False
    bool_s0_unfreezed = False

    best_val_dice = -1.0
    since_best = 0 # counter that counts the number of epochs during which the soft_dice has not improved
    mean_metrics = np.zeros(7, dtype=np.float64)

    new_lr = min_lr

    lr_range_test_file = os.path.join(log_save_path, "lr_range_test.csv")
    csv_file = open(lr_range_test_file, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "lr", "train_loss", "val_loss"])

    scaler = torch.cuda.amp.GradScaler()

    # Dataloader for Validation set
    db_test = SegArtifact_dataset(
        base_dir = config.DATA.DATA_PATH, 
        split = "val", 
        list_dir = config.LIST_DIR,
        transform = transforms.Compose([RandomGenerator(output_size=[img_size, img_size], random_flip_flag = False)]),)
    
    val_loss_loader = DataLoader(
                db_test, 
                batch_size = 1, 
                shuffle = True, 
                num_workers = 1,
                pin_memory = torch.cuda.is_available())
    
    valloader = DataLoader(
                db_test, 
                batch_size = 1, 
                shuffle = False, 
                num_workers = 1,
                pin_memory = torch.cuda.is_available())
    
    some_thing_happend = False # Flag that ensures that when a layer is opened, the optimizer is adjusted accordingly.
    n_layer = 5

    for epoch_num in tqdm(range(max_epoch)):
        model.train()

        for i, g in enumerate(optimizer.param_groups):
            logging.info(f"Group {i}: lr={g['lr']:.3e}, wd={g.get('weight_decay', None)}")
    
        # -------- UNFREEZING THE ENCODER --------
        if freeze_encoder:
            # unfreeze form the deepest encoder level to the highests
            if (epoch_num >= stage3_unfreeze) and (bool_s3_unfreezed == False):
                core(model).unfreeze_encoder(3)
                bool_s3_unfreezed = True
                some_thing_happend = True
                n_layer = 3
            elif (epoch_num >= stage2_unfreeze) and (bool_s2_unfreezed == False):
                core(model).unfreeze_encoder(2)
                bool_s2_unfreezed = True
                some_thing_happend = True
                n_layer = 2
            elif (epoch_num >= stage1_unfreeze) and (bool_s1_unfreezed == False):
                core(model).unfreeze_encoder(1)
                bool_s1_unfreezed = True
                some_thing_happend = True
                n_layer = 1
            elif (epoch_num >= stage0_unfreeze) and (bool_s0_unfreezed == False):
                core(model).unfreeze_encoder(0)
                bool_s0_unfreezed = True
                some_thing_happend = True
                n_layer = 0

            if some_thing_happend == True:
                some_thing_happend = False
                existing_ids = {id(q) for g in optimizer.param_groups for q in g['params']}
                new_params = [p for p in model.parameters()
                            if p.requires_grad and id(p) not in existing_ids]
                if new_params:
                    print(f"Adding {len(new_params)} new parameters to optimizer of layer {n_layer}", file=sys.stderr)
                    base = optimizer.defaults  # enthält u.a. weight_decay, betas, eps
                    wd   = base.get('weight_decay', config.TRAIN.WEIGHT_DECAY)
                    betas= base.get('betas',      config.TRAIN.OPTIMIZER.BETAS)
                    eps  = base.get('eps',        config.TRAIN.OPTIMIZER.EPS)
                    optimizer.add_param_group({
                        'params': new_params,
                        'weight_decay': wd,
                        'betas': betas,
                        'eps': eps,
                        'lr': optimizer.param_groups[0]['lr'],  # konsistent starten
                    })
                else:
                    raise ValueError(f"No new parameter added to optimizer, that's baaad")
        
        # -------------------------------------------
        optimizer.zero_grad(set_to_none=True)
        opt_step = 0
    
        for i_batch, sampled_batch in tqdm(enumerate(trainloader), total=len(trainloader)):
            step = epoch_num*num_batches + i_batch
            # learning rate range test
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            
            is_last_epoch = (epoch_num >= max_epoch - 1)
            is_last_batch = (i_batch == len(trainloader) - 1)

            image_batch = sampled_batch['image'].to(device)
            label_batch = sampled_batch['label'].to(device)
            case_names  = sampled_batch['case_name']
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(image_batch)
            
                # loss
                #loss = tversky_loss(outputs, label_batch)
                #loss = uf_loss(outputs, label_batch)
                loss = dynamic_loss(outputs, label_batch)
            
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            if opt_step % 10 == 0:                
                val_loss = validation_loss(
                    model = model,
                    device = device,
                    val_loader=val_loss_loader,
                    dynamic_loss = dynamic_loss,
                    bool_break = True, # true if you don't want to go through all validation batches, but want to cancel beforehand
                    n_batches = 20, # Only important if bool_break is true. Number of batches to be validated
                    dataset_path = config.DATA.DATA_PATH,
                    list_dir = config.LIST_DIR,
                    img_size = img_size
                    )
                
            opt_step += 1
            lr = optimizer.param_groups[0]['lr']
            csv_writer.writerow([step, lr, loss.item(), val_loss])

            iter_num = iter_num + 1
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            #logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

            """
            # In the last epoch, bit masks and heat masks are created for the last batch.
            if is_last_epoch and is_last_batch:
                logging.info("Last epoch, last batch")
                model.eval()
                with torch.no_grad():
                    create_bin_heat_mask(outputs, case_names, pred_dir, image_batch)
                model.train()
            """
            # add value to learning rate
            new_lr *= mult

        # -------- VALIDATION (aftre every Epoch-Train) --------
        """
        model.eval()
        mean_metrics= inference(
            model = model,
            logging = logging,
            testloader = valloader,
            test_save_path = log_save_path,
            device= device,
            split = "val", 
            img_size = img_size,
            sig_threshold = config.TRAIN.SIG_THRESHOLD)

        mean_dice, mean_IoU, mean_recall, mean_precision, mean_f1_score, mean_soft_dice, mean_soft_IoU = mean_metrics
        write_to_writer(writer, mean_metrics, epoch_num)
        
        if mean_soft_dice > best_val_dice:
            best_val_dice = mean_soft_dice
            since_best = 0
            if config.SAVE_BEST_RUN:
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
            if since_best >= config.TRAIN.EARLY_STOPPING_PATIENCE:
                logging.info(f"Early stopping at epoch {epoch_num} (no val improvement for {config.TRAIN.EARLY_STOPPING_PATIENCE} epochs).")
                break
        """
        # --------------------------------------------------------
        
        # saves all 50 epochs
        save_interval = 50 

        if config.SAVE_BEST_RUN:
            """
            if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
                save_mode_path = os.path.join(log_save_path, 'epoch_' + str(epoch_num) + '.pth')
                torch.save({    'epoch': epoch_num,
                                'model': core(model).state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'iter_num': iter_num,
                                'soft_dice': mean_soft_dice}
                                , save_mode_path,)
                logging.info("save model to {}".format(save_mode_path))
            """

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
                break

        # update learning rate /

        """
        lr_scheduler.step(epoch_num + 1)
        logging.info("epoch:", epoch_num + 1,  " learning rate:", lr_scheduler.get_last_lr())
        """
    csv_file.close()
    writer.close()

    print(f"optimizer steps: {opt_step}", file=sys.stderr)

    csv_reader = pd.read_csv(lr_range_test_file)
    csv_reader["smoothed_train_loss"] = csv_reader["train_loss"].ewm(span=20, adjust=False).mean()
    csv_reader["smoothed_val_loss"] = csv_reader["val_loss"].ewm(span=20, adjust=False).mean()
    plt.figure(figsize=(8, 6))
    plt.plot(csv_reader["lr"], csv_reader["smoothed_train_loss"], label="Smoothed Train Loss", linewidth=2)
    plt.plot(csv_reader["lr"], csv_reader["train_loss"], color='lightblue', alpha=0.3, label="Raw Train Loss")
    plt.plot(csv_reader["lr"], csv_reader["smoothed_val_loss"], color='red', label="Smoothed Validation Loss", linewidth=2)
    plt.plot(csv_reader["lr"], csv_reader["val_loss"], color='salmon', alpha=0.3, label="Raw Validation Loss")
    plt.xscale('log')
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.ylim(0, 2) 
    plt.legend(loc="best")
    plt.title("Learning Rate Range Test For Weight Decay Search")
    plt.grid(True)
    plt.savefig(os.path.join(log_save_path, "weight_decay_test.png"), dpi=300) 
    plt.show()

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
