# test_inference.py
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
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset.dataset import SegArtifact_dataset
from loss.DynamicLoss import DynamicLoss
from network.MSUNet import MSUNet
from scripts.csv_handler import CSV_Handler
from scripts.validation_functions import calculate_metrics
from scripts.map_generator import save_color_heatmap
from dataset.dataset import SegArtifact_dataset, RandomGenerator


def main():

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    parser.add_argument('--check_point_dir', type=str, required=True, metavar="DIR", help='path to directory with best_model.pth')
    args = parser.parse_args()
    check_point_dir = args.check_point_dir

    config = get_config(args, True, False)

    seed = int(config.SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if config.DETERMINISTIC:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False

    # ----- path output ----
    now = datetime.now()
    timestamp_str = now.strftime("%d%m%y_%H%M")
    output_root = os.path.abspath(config.OUTPUT_DIR)
    output_dir  = os.path.join(output_root, f"test_{timestamp_str}")
    os.makedirs(output_dir, exist_ok=True)

    # copy of used config.
    shutil.copy(args.cfg, os.path.join(output_dir, "config_used.yaml"))

    # --- Logger/Writer ---
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        filename=os.path.join(output_dir, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    writer = SummaryWriter(os.path.join(output_dir, 'log'))
    logging.info(f"date: {timestamp_str}")

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- build model ---
    model = MSUNet( config, 
                    img_size = config.DATA.IMG_SIZE, 
                    num_classes=config.MODEL.NUM_CLASSES
                    )
    model.to(device)

    #load checkpoint
    snapshot = os.path.join(check_point_dir, 'best_model.pth')
    if not os.path.exists(snapshot):
        raise FileNotFoundError(f"Checkpoint not found: {snapshot}")
    ckpt = torch.load(snapshot, map_location=device)
    state = None
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state = ckpt['model']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        # evtl. ist es schon direkt ein state_dict
        state = ckpt
    msg = model.load_state_dict(state, strict=True)
    print("loaded checkpoint", msg)

    # --- Dataset/Loader ---
    db_test = SegArtifact_dataset(
        base_dir=config.DATA.DATA_PATH,
        list_dir=config.LIST_DIR,
        split="test",
        transform = transforms.Compose([RandomGenerator(output_size=[config.DATA.IMG_SIZE, config.DATA.IMG_SIZE], random_flip_flag = False, transform = False)])
    )
    test_loader = DataLoader(
        db_test,
        batch_size=1,                   
        shuffle=False,       
        num_workers=1,
        pin_memory=torch.cuda.is_available()
    )

    # --- Loss/Metrik-CSV ---
    dynamic_loss = DynamicLoss(
        alpha=config.TRAIN.TVERSKY_LOSS_ALPHA,
        beta=config.TRAIN.TVERSKY_LOSS_BETA,
        tversky_bce_mix=config.TRAIN.LOSS_TVERSKY_BCE_MIX
    )

    csv_object = CSV_Handler(output_dir)
    (csv_writer,
     csv_batch_fake,
     csv_batch_real,
     csv_real_epoch,
     csv_fake_epoch,
     csv_all_epoch,
     csv_batch) = csv_object.return_writer()

    # --- EVAL ---
    model.eval()

    # --- Metric + Outputs + Test Losss ---
    mean_dice, output_list, Score, FPR = calculate_metrics(
        model=model,
        epoch=1,
        logging=logging,
        testloader=test_loader,
        dynamic_loss=dynamic_loss,
        device=device,
        split="test",     
        img_size=config.DATA.IMG_SIZE,
        sig_threshold=config.TRAIN.SIG_THRESHOLD,
        csv_all_epoch=csv_all_epoch,
        csv_fake_epoch=csv_fake_epoch,
        csv_real_epoch=csv_real_epoch,
        csv_batch_real=csv_batch_real,
        csv_batch_fake=csv_batch_fake,
        mean_train_loss=0.0,
        output_num=len(test_loader)
    )

    # Heatmaps/Masks speichern
    pred_dir = os.path.join(output_dir, "predictions")
    create_bin_heat_mask_from_list(output_list, pred_dir)

    # Zusammenfassung
    logging.info(f"mean_dice_test: {mean_dice:.6f}, Score: {Score:.6f}, FPR: {FPR:.6f}")
    writer.add_scalar('metrics/mean_dice_test', mean_dice, 0)
    writer.add_scalar('metrics/Score_test', Score, 0)
    writer.add_scalar('metrics/FPR_test', FPR, 0)
    writer.close()

    # Für stdout / aufrufende Skripte
    print(timestamp_str, file=sys.stdout)
    return timestamp_str

def create_bin_heat_mask_from_list(output_saver, pred_dir):
    os.makedirs(pred_dir, exist_ok=True)
    for case_name, image_tensor, pred_tensor in output_saver:
        case_name   = str(case_name)
        image_tensor = image_tensor.detach().cpu()
        pred_tensor  = pred_tensor.detach().cpu()

        if pred_tensor.ndim == 4: pred_tensor = pred_tensor[0]
        if image_tensor.ndim == 4: image_tensor = image_tensor[0]

        heat   = pred_tensor.clamp(0, 1)      # [0,1]
        binmsk = (heat > 0.5).float()

        save_image(heat,   os.path.join(pred_dir, f"{case_name}_grey_heats.png"))
        save_image(binmsk, os.path.join(pred_dir, f"{case_name}_bin_mask.png"))

        save_color_heatmap(
            img_3chw=image_tensor,
            heat_hw=heat[0] if heat.ndim == 3 else heat,
            out_png=os.path.join(pred_dir, f"{case_name}_overlay_color.png"),
            alpha=0.45
        )
if __name__ == "__main__":
    ts = main()
