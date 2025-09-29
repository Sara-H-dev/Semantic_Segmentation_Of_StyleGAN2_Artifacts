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
    parser.add_argument('--dataset_path', type=str,
                        default='./dataset', help='root dir for the data')  # for acdc dataset_path=root_dir
    parser.add_argument('--dataset', type=str,
                        default='SegArtifact', help='experiment_name')
    parser.add_argument('--num_classes', type=int,
                        default=1, help='output channel of network')
    parser.add_argument('--list_dir', type=str,
                        default='./lists', help='list dir')
    parser.add_argument('--output_dir', type=str, default='./model_out/timestamp/test', help='output dir')   
    parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='batch_size per gpu')
    parser.add_argument('--img_size', type=int, default=1024, help='input patch size of network input')
    parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
    parser.add_argument('--test_save_dir', type=str, default='.model_out/SegArtifact/test/predictions', help='saving prediction as nii!')
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--sig_threshold', type = float, default = 0.5, help = 'treshold that decides if a pixel is an artefact or not')
    parser.add_argument('--split', type = str, default = 'test',choices=['test', 'val'], help = 'test or val')
    parser.add_argument('--timestamp', type = str, required= True,  help = 'The timestamp from the trainset')

    args = parser.parse_args()

    args.output_dir    = os.path.join('./model_out', args.timestamp, args.split)
    args.test_save_dir = os.path.join(args.output_dir, 'predictions')
    
    config = get_config(args)


    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    args.num_classes = 1
    args.is_pretrain = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MSUNet(config, 
                 img_size = args.img_size, 
                 num_classes = args.num_classes).to(device)
    
    snapshot = os.path.join(args.output_dir, 'best_model.pth')
    if not os.path.exists(snapshot):
        snapshot = os.path.join(args.output_dir, f'epoch_{args.max_epochs-1}.pth')
    if not os.path.exists(snapshot):
        raise FileNotFoundError(f"Checkpoint not found: {snapshot}")
    ckpt = torch.load(snapshot, map_location=device)
    msg = model.load_state_dict(ckpt['model'], strict=True)
    print("self trained ms_unet",msg)
    snapshot_name = snapshot.split('/')[-1]

    log_folder = os.path.join('./model_out', args.timestamp, args.split, 'test_log')
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

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir 
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(model, logging, test_save_path, device, args.dataset_path, 
                args.split, args.list_dir, args.img_size, args.sig_threshold)

if __name__ == "__main__":
    main()
    


