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

def main():
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument('--root_path', type=str, default='./dataset', help='root dir for data')
    parser.add_argument('--output_dir', type=str, default='./model_out', help='output dir') 
    parser.add_argument('--list_dir', type=str, default='./lists', help='list dir')
    # Hyperparameters
    parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')
    parser.add_argument('--max_epochs', type=int, default=30000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int, default=1024, help='input patch size of network input')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    # forces the pythorch/CUDA to just perform deterministic operation, so the exact same result is archieved every time
    parser.add_argument('--deterministic', action='store_true', help='use deterministic training')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    # parameters of the tversky loss
    parser.add_argument('--loss_alpha', type = float, default = 0.4, help='parameter for the tversky loss')
    parser.add_argument('--loss_beta', type = float, default = 0.6, help='parameter for the tversky loss')
    # if encoder should be frozen:
    parser.add_argument('--freeze_encoder', type = bool, default = True, help='If true, encoder is frozen')
    # percent of epochs then the stages are unfrozen:
    parser.add_argument('--unfreeze_stage3', type = float, default = 0.4, help='Percentage of epochs when this stage is unfrozen')
    parser.add_argument('--unfreeze_stage2', type = float, default = 0.7, help='Percentage of epochs when this stage is unfrozen')
    parser.add_argument('--unfreeze_stage1', type = float, default = 0.9, help='Percentage of epochs when this stage is unfrozen')
    parser.add_argument('--unfreeze_stage0', type = float, default = 0.98, help='Percentage of epochs when this stage is unfrozen')
    # threshold for the gereration of the binary mask for the validation
    parser.add_argument('--sig_threshold', type = float, default = 0.5, help = 'treshold that decides if a pixel is an artefact or not')
    # number of epochs at which the process is terminated if the result does not improve
    parser.add_argument('--early_stopping_patience', type = int, default = 15, help = 'number of epochs at which the process is terminated if the result does not improve')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )

    args = parser.parse_args()
    now = datetime.now()
    # format: DayMonthYear_HourMinute
    timestamp_str = now.strftime("%d%m%y_%H%M")
    args.output_dir = os.path.join(args.output_dir, timestamp_str)

    config = get_config(args)
    # checks if the training should be deterministic or not
    if not args.deterministic:
        cudnn.benchmark = True 
        cudnn.deterministic = False 
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # calculation of base learning rate
    ref_bs = 24
    acc_steps = args.accumulation_steps or 1
    effective_bs = args.batch_size * acc_steps
    if effective_bs != ref_bs:
        logging.info(f"[LR] Scaling base_lr {args.base_lr:.6f} -> {args.base_lr * (effective_bs/ref_bs):.6f} "
            f"(eff_BS={effective_bs}, ref={ref_bs})")
        args.base_lr *= effective_bs / ref_bs

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model = MSUNet( logging, config, 
                    img_size=args.img_size, 
                    num_classes=args.num_classes
                    )
    # pretrained weights are loaded
    try:
        model.load_segface_weight(config)
    except Exception as e:
        logging.error(f"Could not load segface weights: {e}")

    # if cuda is avilable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # train dictionary wiht the trianer_MS_UNet function
    trainer_dic = {'SegArtifact': trainer,}
    trainer_dic['SegArtifact'](args, model, args.output_dir, config)
    return timestamp_str

if __name__ == "__main__":
    timestamp_str = ""
    timestamp_str = main()
    print(timestamp_str, file=sys.stdout)