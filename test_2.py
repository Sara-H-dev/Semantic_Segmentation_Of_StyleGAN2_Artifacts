# 
import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import sys
import shutil
 
from config import get_config
from datetime import datetime
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from torch.utils.data import DataLoader, ConcatDataset, Subset
from tqdm import tqdm

from dataset.dataset import SegArtifact_dataset, RandomGenerator
from loss.DynamicLoss import DynamicLoss
from network.MSUNet import MSUNet
from scripts.csv_handler import CSV_Handler
from scripts.validation_functions import calculate_metrics
from scripts.map_generator import save_color_heatmap

def main():
    torch.backends.cudnn.benchmark = False   # vermeidet riesige cuDNN-Workspaces
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument('--check_point_dir', type=str, required=True, metavar="FILE", help='path to best_model.pth', )

    args = parser.parse_args()
    check_point_dir = args.check_point_dir

    config = get_config(args, True, False)

    now = datetime.now()
    # format: DayMonthYear_HourMinute
    timestamp_str = now.strftime("%d%m%y_%H%M")
    print(f"tima: {timestamp_str}")
    output_dir = os.path.join(config.OUTPUT_DIR, timestamp_str)
    # output_dir = os.path.join(config.OUTPUT_DIR, timestamp_str)
    output_dir = config.OUTPUT_DIR
    seed = config.SEED
    batch_size = config.DATA.BATCH_SIZE
    base_lr = config.TRAIN.BASE_LR
    img_size = config.DATA.IMG_SIZE
    num_classes = config.MODEL.NUM_CLASSES
    print(f"Weight_decay = {config.TRAIN.WEIGHT_DECAY}")
    print(f"Drop_path = {config.MODEL.DROP_PATH_RATE}")
    print(f"Drop_rate = {config.MODEL.DROP_RATE}")
    print(f"Attention Drop = {config.MODEL.ATTN_DROP_RATE}")
    print(f"tversky alpha = {config.TRAIN.TVERSKY_LOSS_ALPHA}")
    print(f"tversky beta = {config.TRAIN.TVERSKY_LOSS_BETA}")
    print(f"tversky_bce_mix_factor = { config.TRAIN.LOSS_TVERSKY_BCE_MIX}")
    print(f"base_lr = {config.TRAIN.BASE_LR}")
    print(f"Dynamic_LOADER = {config.Dynamic_LOADER}")
    print(f"warm_up = {config.TRAIN.WARMUP_EPOCHS}")
    print(f"epochs = {config.TRAIN.MAX_EPOCHS}")
    print(f"seed = {config.SEED}")
    print(f"pretrained weights = {config.MODEL.PRETRAIN_WEIGHTS}")

   
    os.makedirs(output_dir, exist_ok=True)
    # copy the yaml to model_out
    config_path = args.cfg
    shutil.copy(config_path, os.path.join(output_dir, "config_used.yaml"))

    # checks if the training should be deterministic or not
    if not config.DETERMINISTIC:
        cudnn.benchmark = True 
        cudnn.deterministic = False 
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    # ---------- logger ------------
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename = os.path.join(output_dir, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', 
        datefmt='%H:%M:%S') #houres, minutes, seconds
    writer = SummaryWriter(output_dir + '/log')

    timestamp_str = now.strftime("%d%m%y_%H%M")
    logging.info(f"date: {timestamp_str}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MSUNet( config, 
                    img_size = img_size, 
                    num_classes = num_classes
                    )
    
    snapshot = os.path.join(check_point_dir, 'best_model.pth')
    if not os.path.exists(snapshot):
        raise FileNotFoundError(f"Checkpoint not found: {snapshot}")
    ckpt = torch.load(snapshot, map_location=device)
    msg = model.load_state_dict(ckpt['model'], strict=True)
    print("loaded checkpoint",msg)
    snapshot_name = snapshot.split('/')[-1]
    
    db_test = SegArtifact_dataset( base_dir = config.DATA.DATA_PATH, 
                                    list_dir = config.LIST_DIR, 
                                    split = "test",
                                    transform = None)
    
    test_loader = DataLoader(
                db_test, 
                batch_size = 1, 
                shuffle = True, 
                num_workers = 1,
                pin_memory = torch.cuda.is_available())
    
    dynamic_loss = DynamicLoss(alpha=config.TRAIN.TVERSKY_LOSS_ALPHA, beta=config.TRAIN.TVERSKY_LOSS_BETA, tversky_bce_mix = config.TRAIN.LOSS_TVERSKY_BCE_MIX)

    csv_object = CSV_Handler(output_dir)
    csv_writer, csv_batch_fake, csv_batch_real, csv_real_epoch, csv_fake_epoch, csv_all_epoch, csv_batch = csv_object.return_writer()
    
    test_loss_list =[]
    model.eval() 

    for i_batch, sampled_batch in tqdm(enumerate(test_loader), total=len(test_loader)):

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
            loss = dynamic_loss(outputs, label_batch)
            test_loss_list.append(loss.item())

    mean_test_loss = sum(test_loss_list) / len(test_loss_list)

    mean_dice, output_dict, Score, FPR = calculate_metrics(
            model = model,
            epoch = 1,
            logging = logging,
            testloader = test_loader,
            dynamic_loss = dynamic_loss,
            device= device,
            split = "val", 
            img_size = img_size,
            sig_threshold = config.TRAIN.SIG_THRESHOLD,
            csv_all_epoch = csv_all_epoch,
            csv_fake_epoch = csv_fake_epoch,
            csv_real_epoch = csv_real_epoch,
            csv_batch_real = csv_batch_real,
            csv_batch_fake = csv_batch_fake,
            mean_train_loss = 0,
            output_num = len(test_loader))
    
    create_bin_heat_mask_from_list(output_dict, output_dir)

    return timestamp_str

def create_bin_heat_mask_from_list(output_saver, pred_dir):

    for case_name, image_tensor, pred_tensor in output_saver:
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


if __name__ == "__main__":
    timestamp_str = ""
    timestamp_str = main()
    print(timestamp_str, file=sys.stdout)