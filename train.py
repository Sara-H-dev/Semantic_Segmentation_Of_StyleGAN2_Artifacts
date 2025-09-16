# 
import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from network.MSUNet import MSUNet as MSUNet
from trainer import trainer_MS_UNet
from config import get_config

parser = argparse.ArgumentParser()
# path to the dataset
parser.add_argument('--root_path', type=str,
                    default='./datasets/SegArtifact', help='root dir for data')
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
parser.add_argument('--output_dir', type=str, default='./model_out/SegArtifact', help='output dir') 
# max iteramtion               
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
# max epoch 
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
# batch size
parser.add_argument('--batch_size', type=int,
                    default=6, help='batch_size per gpu')
# numer of used GPUs
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
# forces the pythorch/CUDA to just perform deterministic operation
# so the exact same result is archieved every time
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
# learning rate
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
# image size (BxW)
parser.add_argument('--img_size', type=int,
                    default=1024, help='input patch size of network input')
# random seed (set for reproductioon)
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
# path to JSON with model/train/eval-setings
parser.add_argument('--cfg', type=str, 
                    required=True, metavar="FILE", help='path to config file', )
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
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
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

args = parser.parse_args()
if args.dataset == "SegArtifact":
    args.root_path = os.path.join(args.root_path, "train_npz")
config = get_config(args)


if __name__ == "__main__":
    # checks if the training should be deterministic or not
    if not args.deterministic:
        cudnn.benchmark = True 
        cudnn.deterministic = False 
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'SegArtifact': {
            'root_path': args.root_path,
            'list_dir': './lists',
            'num_classes': 1,
        },
    }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # .cuda() puts the model on GPU, everything on VRAM
    model = MSUNet( config, 
                    img_size=args.img_size, 
                    num_classes=args.num_classes
                    ).cuda()
    # pretrained weights are loaded
    model.load_from(config)

    # train dictionary wiht the trianer_MS_UNet function
    trainer_dic = {'SegArtifact': trainer_MS_UNet,}
    trainer_dic[dataset_name](args, model, args.output_dir)