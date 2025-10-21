# 
import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import sys
from network.MSUNet import MSUNet 
from trainer import trainer
from config import get_config
from datetime import datetime
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./model_out', help='output dir') 
    parser.add_argument('--deterministic', action='store_true', help='use deterministic training')
    parser.add_argument('--use_checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--sig_threshold', type = float, help = 'treshold that decides if a pixel is an artefact or not')
    #parser.add_argument('--weight_decay', type = float)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )

    args = parser.parse_args()

    config = get_config(args, True, False)

    now = datetime.now()
    # format: DayMonthYear_HourMinute
    timestamp_str = now.strftime("%d%m%y_%H%M")
    output_dir = os.path.join(config.OUTPUT_DIR, timestamp_str, str(config.TRAIN.WEIGHT_DECAY))
    # output_dir = os.path.join(config.OUTPUT_DIR, timestamp_str)
    seed = config.SEED
    batch_size = config.DATA.BATCH_SIZE
    base_lr = config.TRAIN.BASE_LR
    img_size = config.DATA.IMG_SIZE
    num_classes = config.MODEL.NUM_CLASSES
    print(f"Weight_decay = {config.TRAIN.WEIGHT_DECAY}")
    
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

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = MSUNet( config, 
                    img_size = img_size, 
                    num_classes = num_classes
                    )
    
    # pretrained weights are loaded
    try:
        if config.MODEL.PRETRAIN_WEIGHTS == 'segface':
            model.load_segface_weight(config)
        elif config.MODEL.PRETRAIN_WEIGHTS == 'imagenet1k':
            model.load_IMAGENET1K_weight(config)
        else:
            raise ValueError(f"Could not load pretrained weights")
    except Exception as e:
        raise ValueError(f"Could not load pretrained weights: {e}")

    # if cuda is avilable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # train dictionary wiht the trianer_MS_UNet function
    trainer_dic = {'SegArtifact': trainer,}
    trainer_dic['SegArtifact'](model, output_dir, config, base_lr)
  
    return timestamp_str

if __name__ == "__main__":
    timestamp_str = ""
    timestamp_str = main()
    print(timestamp_str, file=sys.stdout)