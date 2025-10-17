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
from datetime import datetime

from tqdm import tqdm
from loss.DynamicLoss import DynamicLoss
from scripts.map_generator import save_color_heatmap
from scripts.validation_functions import calculate_metrics, validation_loss
from scripts.csv_handler import CSV_Handler

def trainer(model, log_save_path = "", config = None, base_lr = 5e-4):
    from dataset.dataset import SegArtifact_dataset, RandomGenerator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ----------- get from config ---------
    if config == None: 
        logging.error("Config file is not found!")
        raise ValueError("Config file is not found!")
    
    warmup_epochs = config.TRAIN.WARMUP_EPOCHS
    max_epoch = config.TRAIN.MAX_EPOCHS
    freeze_encoder = config.MODEL.FREEZE_ENCODER 
    n_gpu = config.HARDWARE.N_GPU
    img_size = config.DATA.IMG_SIZE
    batch_size = config.DATA.BATCH_SIZE
    if n_gpu > 0:
        batch_size = batch_size * n_gpu   # batch_size
    #--------------------------------------
    
    # folder for the final output of binary and heatmap masks
    os.makedirs(log_save_path, exist_ok=True)
    pred_dir = os.path.join(log_save_path, "final_preds")
    os.makedirs(pred_dir, exist_ok=True)

    # ---------- logger and csv config ------------
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename = os.path.join(log_save_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', 
        datefmt='%H:%M:%S') #houres, minutes, seconds
    writer = SummaryWriter(log_save_path + '/log')

    csv_object = CSV_Handler(log_save_path)
    csv_writer, csv_batch_fake, csv_batch_real, csv_real_epoch, csv_fake_epoch, csv_all_epoch = csv_object.return_writer()
    val_loss = float("nan")

    # ----------- preparation of training data ------------------------
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
    num_batches = len(trainloader)
    print(f"{num_batches} iterations per epoch.", file=sys.stderr)

    # ----------------- Dataloader for Validation set ---------------
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
    # ------------- Loss Function --------------------------------
    dynamic_loss = DynamicLoss(alpha=config.TRAIN.TVERSKY_LOSS_ALPHA, beta=config.TRAIN.TVERSKY_LOSS_BETA)

    # ---------------------- freeze encoder ------------------------------
    def core(m):  
        return m.module if isinstance(m, nn.DataParallel) else m   
    core(model).freeze_encoder(freeze_encoder) # freez encoder if wanted
    
    # ------------- AdamW Optimizer ------------------------------
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = base_lr,
        betas= config.TRAIN.OPTIMIZER.BETAS,   # "Momentum"-Parameter
        eps = config.TRAIN.OPTIMIZER.EPS,             # kleine Konstante für Stabilität
        weight_decay = config.TRAIN.WEIGHT_DECAY,    # L2-Regularisierung (entkoppelt!)
        amsgrad = False         # optional, selten genutzt
    )

    # ------------- Cosine Decay with linear warmup ---------------
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial= max_epoch - warmup_epochs,
        lr_min = config.TRAIN.MIN_LR,
        warmup_lr_init = config.TRAIN.WARMUP_LR,
        warmup_t = warmup_epochs,
        cycle_limit = 1,
        t_in_epochs=  True,
        warmup_prefix = config.TRAIN.LR_SCHEDULER.WARMUP_PREFIX
    )
                                                    
    # ----------- parameters for unfreezing the encoder --------------------
    stage3_unfreeze = int(max_epoch * config.MODEL.STAGE3_UNFREEZE_PERIODE); bool_s3_unfreezed = False
    stage2_unfreeze = int(max_epoch * config.MODEL.STAGE2_UNFREEZE_PERIODE); bool_s2_unfreezed = False
    stage1_unfreeze = int(max_epoch * config.MODEL.STAGE1_UNFREEZE_PERIODE); bool_s1_unfreezed = False
    stage0_unfreeze = int(max_epoch * config.MODEL.STAGE0_UNFREEZE_PERIODE); bool_s0_unfreezed = False
    
    # ---------------------- Flags an inital values -------------------------------------
    best_val_dice = -1.0 # stores the best value of the mean soft dice for the validation set
    since_best = 0 # counter that counts the number of epochs during which the soft_dice has not improved
    iter_num = 0
    best_performance = 0.0
    scaler = torch.cuda.amp.GradScaler() # scaler to improve speed
    some_thing_happend = False # Flag that ensures that when a layer is opened, the optimizer is adjusted accordingly.
    n_layer = 5
    last_run = False
    opt_step = 0
    unfreeze_in_next_epoch = False

    

    # ====================== Training START ==========================================

    for epoch_num in tqdm(range(max_epoch)):
    
        # -------- UNFREEZING THE ENCODER --------
        print(f"freeze encoder is {freeze_encoder}")
        if freeze_encoder:
            # unfreeze form the deepest encoder level to the highests
            # if stage_unfreeze is triggered, or where has not been a improvment for config.TRAIN.EARLY_STOPPING_PATIENCE
            if ((epoch_num >= stage3_unfreeze) or unfreeze_in_next_epoch == True) and (bool_s3_unfreezed == False):
                core(model).unfreeze_encoder(3); bool_s3_unfreezed = True; n_layer = 3 ; some_thing_happend = True; unfreeze_in_next_epoch == False
            elif ((epoch_num >= stage2_unfreeze) or unfreeze_in_next_epoch == True) and (bool_s2_unfreezed == False):
                core(model).unfreeze_encoder(2) ; bool_s2_unfreezed = True; n_layer = 2 ; some_thing_happend = True; unfreeze_in_next_epoch == False
            elif ((epoch_num >= stage1_unfreeze)or unfreeze_in_next_epoch == True) and (bool_s1_unfreezed == False):
                core(model).unfreeze_encoder(1); bool_s1_unfreezed = True; n_layer = 1 ; some_thing_happend = True ; unfreeze_in_next_epoch == False
            elif ((epoch_num >= stage0_unfreeze)or unfreeze_in_next_epoch == True) and (bool_s0_unfreezed == False):
                core(model).unfreeze_encoder(0); bool_s0_unfreezed = True; n_layer = 0 ; some_thing_happend = True; unfreeze_in_next_epoch == False 
                
            if some_thing_happend: # --- after an unfreazing the optimizer needs to be updated
                some_thing_happend = False
                existing_ids = {id(q) for g in optimizer.param_groups for q in g['params']}
                new_params = [p for p in model.parameters()
                    if p.requires_grad and id(p) not in existing_ids]

                if new_params:
                    
                    base = optimizer.defaults  # contains weight_decay, betas, eps
                    wd   = base.get('weight_decay', config.TRAIN.WEIGHT_DECAY)
                    betas= base.get('betas',      config.TRAIN.OPTIMIZER.BETAS)
                    eps  = base.get('eps',        config.TRAIN.OPTIMIZER.EPS)
                    lr    = optimizer.param_groups[0]['lr']

                    optimizer.add_param_group({ 
                        'params': new_params,  
                        'weight_decay': wd,
                        'betas': betas, 
                        'eps': eps, 
                        'lr': lr }) 
                    print(f"Adding {len(new_params)} new parameters to optimizer of layer {n_layer}")
                else:
                    raise ValueError(f"No new parameter added to optimizer  {n_layer}, that's baaad")
        
        # -------------------------------------------
        for i, g in enumerate(optimizer.param_groups):
            logging.info(f"Group {i}: lr={g['lr']:.3e}, wd={g.get('weight_decay', None)}")
        optimizer.zero_grad(set_to_none=True)

        # ==================== TRAIN EPOCH ======================================
        for i_batch, sampled_batch in tqdm(enumerate(trainloader), total=len(trainloader)):
            model.train() # training modus
            step = epoch_num*num_batches + i_batch

            image_batch = sampled_batch['image'].to(device)
            label_batch = sampled_batch['label'].to(device)
            case_names  = sampled_batch['case_name']
            
            """
            # learning rate range test
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr """
            # ------------- training ------------------
            with torch.amp.autocast('cuda', dtype=torch.float16):

                outputs = model(image_batch) # prediction
                loss = dynamic_loss(outputs, label_batch)  # loss
            
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                opt_step += 1
                lr = optimizer.param_groups[0]['lr']
            
            # ------------ calculating validation loss ---------
            if opt_step % 100 == 0:                
                val_loss = validation_loss(
                    model = model,
                    device = device,
                    val_loader=val_loss_loader,
                    dynamic_loss = dynamic_loss,
                    bool_break = True, # true if you don't want to go through all validation batches, but want to cancel beforehand
                    n_batches = 20, # Only important if bool_break is true. Number of batches to be validated
                    )

            # ------------------- logging --------------------------   
            csv_writer.writerow([step, lr, loss.item(), val_loss])
            iter_num = iter_num + 1;  writer.add_scalar('info/total_loss', loss.item(), iter_num)
        #==================== END OF EPOCH ======================================

        # -------- VALIDATION ----------------
        model.eval()
        mean_dice, ten_output_dict = calculate_metrics(
            model = model,
            epoch = epoch_num + 1,
            logging = logging,
            testloader = valloader,
            dynamic_loss = dynamic_loss,
            device= device,
            split = "val", 
            img_size = img_size,
            sig_threshold = config.TRAIN.SIG_THRESHOLD,
            csv_all_epoch = csv_all_epoch,
            csv_fake_epoch = csv_fake_epoch,
            csv_real_epoch = csv_real_epoch,
            csv_batch_real = csv_batch_real,
            csv_batch_fake = csv_batch_fake)
        
        # ------------------ save best run --------------------------------
        if mean_dice > best_val_dice:
            save_best_output = ten_output_dict
            best_val_dice = mean_dice
            since_best = 0
            if config.SAVE_BEST_RUN:
                best_path = os.path.join(log_save_path, 'best_model.pth')
                torch.save({ 'epoch': epoch_num, 'model': core(model).state_dict(), 'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num, 'best_val_dice': best_val_dice }, best_path)
                logging.info(f"Saved new BEST checkpoint to {best_path} (val_dice={best_val_dice:.5f})")
        else: # ----------- early stopping -------------------------------
            since_best += 1
            if since_best >= config.TRAIN.EARLY_STOPPING_PATIENCE:
                if bool_s0_unfreezed == True:
                    # if all encoder layers are unfreezed and where is no improvmetn
                    # early stopping is applied
                    logging.info(f"Early stopping at epoch {epoch_num} (no val improvement for {config.TRAIN.EARLY_STOPPING_PATIENCE} epochs).")
                    last_run = True
                else:
                    # in the next run the next encoder layer is unfreezed
                    unfreeze_in_next_epoch = True
                
        
        # -------------------- saves the last run ------------------------------------
        if epoch_num >= max_epoch - 1:
            last_run = True
            if config.SAVE_LAST_RUN: 
                save_mode_path = os.path.join(log_save_path, 'epoch_' + str(epoch_num) + '.pth')
                torch.save({'epoch': epoch_num, 'model':  core(model).state_dict(), 'optimizer': optimizer.state_dict(),       
                            'iter_num': iter_num, 'dice': mean_dice}, save_mode_path)

        # --------------------- update learning rate ---------------------------------
        lr_scheduler.step(epoch_num + 1)
        current_lrs = [g['lr'] for g in optimizer.param_groups]
        logging.info(f"epoch {epoch_num + 1} | lrs={[f'{lr:.3e}' for lr in current_lrs]}")

        # --------------------- if last epoch ----------------------------------------
        if last_run == True:
            create_bin_heat_mask_from_list(save_best_output, pred_dir)
            break

    csv_object.close_files()
    writer.close()
    now = datetime.now()
    timestamp_str_after = now.strftime("%d%m%y_%H%M")
    logging.info(f"Finised at time: {timestamp_str_after}")
    return "Training Finished!"

# ---------------- plot learning range ------------------
def plot_lr_range(lr_range_test_file, log_save_path):
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

# ------------ sets seeds --------------------
def make_worker_init_fn(base_seed: int):
    def _init(worker_id: int):
        seed = base_seed + worker_id
        random.seed(seed); 
        np.random.seed(seed); 
        torch.manual_seed(seed)
    return _init

# ------------ creates heat mask and bin mask --------------------
def create_bin_heat_mask_from_list(ten_output_saver, pred_dir):

    for case_name, image_tensor, pred_tensor in ten_output_saver:
        case_name = str(case_name)
        image_tensor = image_tensor.detach().cpu()
        pred_tensor  = pred_tensor.detach().cpu()

        if pred_tensor.ndim == 4: pred_tensor = pred_tensor[0] 
        if image_tensor.ndim == 4: image_tensor = image_tensor[0]
            
        heat   = pred_tensor.clamp(0, 1)         # in [0,1]
        binmsk = (heat > 0.5).float()

        save_image(heat, os.path.join(pred_dir, f"{case_name}_grey_heats.png"))
        save_image(binmsk, os.path.join(pred_dir, f"{case_name}_bin_mask.png"))

        save_color_heatmap(
            img_3chw=image_tensor,
            heat_hw=heat[0] if heat.ndim == 3 else heat,
            out_png=os.path.join(pred_dir, f"{case_name}_overlay_color.png"),
            alpha= 0.45 )


    
