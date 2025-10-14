import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tqdm import tqdm
from dataset.dataset import SegArtifact_dataset
from scripts.inference import inference
from network.MSUNet import MSUNet
from config import get_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='output dir')   
    parser.add_argument('--deterministic', action="store_true", help='whether use deterministic training')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--use_checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--sig_threshold', type = float,  help = 'treshold that decides if a pixel is an artefact or not')
    parser.add_argument('--split', type = str, default = 'test',required= True, choices=['test', 'val'], help = 'test or val')
    parser.add_argument('--timestamp', type = str, required= True,  help = 'The timestamp from the trainset')

    args = parser.parse_args()

    output_dir = os.path.join(config.OUTPUT_DIR, args.timestamp, args.split)
    
    config = get_config(args, False, True)


    if not config.DETERMINISTIC:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)

    num_classes = config.MODEL.NUM_CLASSES
    output_dir = config.OUTPUT_DIR
    max_epochs = config.TRAIN.MAX_EPOCHS 
    split = args.split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MSUNet(config, 
                 img_size = config.DATA.IMG_SIZE, 
                 num_classes = num_classes).to(device)
    
    snapshot = os.path.join(output_dir, 'best_model.pth')
    if not os.path.exists(snapshot):
        snapshot = os.path.join(output_dir, f'epoch_{max_epochs - 1}.pth')
    if not os.path.exists(snapshot):
        raise FileNotFoundError(f"Checkpoint not found: {snapshot}")
    ckpt = torch.load(snapshot, map_location=device)
    msg = model.load_state_dict(ckpt['model'], strict=True)
    print("self trained ms_unet",msg)
    snapshot_name = snapshot.split('/')[-1]

    log_folder = os.path.join('./model_out', args.timestamp, split, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        filename = os.path.join(log_folder, f"{snapshot_name}.txt"),
        level = logging.INFO, 
        format = '[%(asctime)s.%(msecs)03d] %(message)s', 
        datefmt = '%H:%M:%S',
        force = True)
    
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    test_save_dir = os.path.join(output_dir, "predictions")
    if config.TEST.IS_SAVENII:
        test_save_path = test_save_dir 
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(model, logging, test_save_path, device, config.DATA.DATA_PATH, 
                split, config.LIST_DIR, config.DATA.IMG_SIZE, config.TEST.SIG_THRESHOLD)

if __name__ == "__main__":
    main()
    


