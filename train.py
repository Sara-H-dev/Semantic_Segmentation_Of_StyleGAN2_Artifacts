# 
import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from network.MSUNet import MSUNet 
from trainer import trainer
from config import get_config
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    # path to the dataset
    parser.add_argument('--root_path', type=str,
                        default='./dataset', help='root dir for data')
    # name of the experiment or dataset
    parser.add_argument('--dataset', type=str,
                        default='SegArtifact', help='experiment_name')
    # folder to the data list
    parser.add_argument('--list_dir', type=str,
                        default='./lists', help='list dir')
    # number of classes
    parser.add_argument('--num_classes', type=int,
                        default=1, help='output channel of network')
    # folder for logs, checkpoints, preds
    parser.add_argument('--output_dir', type=str, default='./model_out', help='output dir') 
    # max epoch 
    parser.add_argument('--max_epochs', type=int,
                        default=30000, help='maximum epoch number to train')
    # batch size
    parser.add_argument('--batch_size', type=int,
                        default=2, help='batch_size per gpu')
    # numer of used GPUs
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    # forces the pythorch/CUDA to just perform deterministic operation
    # so the exact same result is archieved every time
    parser.add_argument('--deterministic', action='store_true', help='use deterministic training')
    # learning rate
    parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    # image size (BxW)
    parser.add_argument('--img_size', type=int,
                        default=1024, help='input patch size of network input')
    # random seed (set for reproductioon)
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    # Modify config options by adding 'KEY VALUE' pairs.
    parser.add_argument(
            "--opts",
            help="Modify config options by adding 'KEY VALUE' pairs. ",
            default=None,
            nargs='+',
        )
    # if you want to load the data out of zips not our of folders
    parser.add_argument('--zip', action='store_true', 
                        help='use zipped dataset instead of folder dataset')
    # catching strategie
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                                'full: cache all data, '
                                'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    # path to load checkpoint
    parser.add_argument('--resume', help='resume from checkpoint')
    # calucaltes mini-batces and calculates gradient, and then doese opimization
    # if you want batch=32 but your GPU just can handle 8
    # -> set accumulation-steps = 4
    parser.add_argument('--accumulation-steps', default=4, type=int, help="gradient accumulation steps")
    # if you want to use ceckpoints
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    # AMP = Automatic Mixed Precision
    # technick to make training faster and storage efficent
    # calculates not 32- Bit but 16-Bit
    # disadvantage: can be numerical instabel
    # O0: no AMP
    # O1: Mixed Precision (both 32- and 16-Bit)
    # O2: Nearly everything 16-Bit
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    # tag of experiment
    parser.add_argument('--tag', help='tag of experiment')
    # no training just evaluation
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    # just for erfomance test (how much pictures can my model process per second?)
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
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
    parser.add_argument('--early_stopping_patience ', type = int, default = 15, help = 'number of epochs at which the process is terminated if the result does not improve')

 
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

    dataset_name = args.dataset
    dataset_config = {
        'SegArtifact': {
            'root_path': args.root_path,
            'list_dir': './lists',
            'num_classes': args.num_classes,
        },
    }

    # calculation of base learning rate
    ref_bs = 24
    acc_steps = args.accumulation_steps or 1
    effective_bs = args.batch_size * acc_steps
    if effective_bs != ref_bs:
        print(f"[LR] Scaling base_lr {args.base_lr:.6f} -> {args.base_lr * (effective_bs/ref_bs):.6f} "
            f"(eff_BS={effective_bs}, ref={ref_bs})")
        args.base_lr *= effective_bs / ref_bs

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model = MSUNet( config, 
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
    # model.eval()

    # train dictionary wiht the trianer_MS_UNet function
    trainer_dic = {'SegArtifact': trainer,}
    trainer_dic['SegArtifact'](args, model, args.output_dir, config)


if __name__ == "__main__":
    main()