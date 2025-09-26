import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.dataset import SegArtifact_dataset
from scripts.predict_and_score import predict_and_score
from network.MSUNet import MSUNet as MSUNet
from config import get_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str,
                        default='./dataset', help='root dir for the data')  # for acdc test_path=root_dir
    parser.add_argument('--dataset', type=str,
                        default='SegArtifact', help='experiment_name')
    parser.add_argument('--num_classes', type=int,
                        default=1, help='output channel of network')
    parser.add_argument('--list_dir', type=str,
                        default='./lists', help='list dir')
    parser.add_argument('--output_dir', type=str, default='./model_out/timestamp/test', help='output dir')   
    parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='batch_size per gpu')
    parser.add_argument('--img_size', type=int, default=1024, help='input patch size of network input')
    parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
    parser.add_argument('--test_save_dir', type=str, default='.model_out/SegArtifact/test/predictions', help='saving prediction as nii!')
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--cfg', type=str, 
                        default = 'None', required=False, metavar="FILE", help='path to config file', )
    parser.add_argument(
            "--opts",
            help="Modify config options by adding 'KEY VALUE' pairs. ",
            default=None,
            nargs='+',
        )
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                                'full: cache all data, '
                                'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--sig_threshold', type = float, default = 0.5, help = 'treshold that decides if a pixel is an artefact or not')
    parser.add_argument('--split', type = str, default = 'test',choices=['test', 'val'], help = 'test or val')
    parser.add_argument('--timestamp', type = str, required= True,  help = 'The timestamp from the trainset')

    args = parser.parse_args()
    if args.dataset == "SegArtifact":
        args.test_path = os.path.join(args.test_path)

    args.output_dir = './model_out/' + args.timestamp + '/' + args.split
    args.test_save_dir = './model_out/' + args.timestamp + '/' + args.split + '/predictions'
    
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

    dataset_config = {
        'SegArtifact': {
            'Dataset': SegArtifact_dataset,
            'test_path': args.test_path,
            'list_dir': './lists',
            'num_classes': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.test_path = dataset_config[dataset_name]['test_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MSUNet(config, 
                 img_size = args.img_size, 
                 num_classes = args.num_classes).to(device)

    snapshot = os.path.join(args.output_dir, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    msg = model.load_state_dict(torch.load(snapshot), map_location = device)
    print("self trained ms_unet",msg)
    snapshot_name = snapshot.split('/')[-1]

    log_folder = './model_out/' + args.timestamp + 'test_log'
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
    inference(args, model, test_save_path, device)


def inference(args, model, test_save_path=None, device = None):

    db_test = args.Dataset(
            base_dir = args.test_path, 
            split = args.split, 
            list_dir = args.list_dir)
    
    testloader = DataLoader(
            db_test, 
            batch_size = 1, 
            shuffle = False, 
            num_workers = 1)
    
    logging.info("{} test iterations per epoch".format(len(testloader)))

    model.eval()

    num_cases = 0

    metrics_sum = np.zeros(5, dtype=np.float64)  # [dice, IoU, recall, precision, f1]

    for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader)):
        image = sampled_batch["image"]
        label = sampled_batch["label"]
        case_name =  sampled_batch['case_name'][0]

        metric_i = predict_and_score(image, 
                                      label, 
                                      model,  
                                      patch_size = [args.img_size, args.img_size],
                                      test_save_path = test_save_path, 
                                      case = case_name, 
                                      device = device,
                                      threshold = args.sig_threshold)
        
        # Transfer to a robust 1D vector and limit to 5 key figures
        metric_i = np.asarray(metric_i, dtype=np.float64).reshape(-1)[:5]

        
        if metric_i.shape[0] != 5:
            msg = f"Expected 5 metrics, got {metric_i.shape[0]} for case {case_name}"
            logging.error(msg)
            raise ValueError(msg)

        metrics_sum += metric_i
        num_cases += 1

        m_dice, m_iou, m_rec, m_prec, m_f1 = metric_i

        logging.info(
            f"idx {i_batch} case {case_name} "
            f"mean_dice {m_dice:.4f} mean_IoU {m_iou:.4f} "
            f"mean_recall {m_rec:.4f} mean_precision {m_prec:.4f} mean_f1_score {m_f1:.4f}"
        )
    
    if num_cases == 0:
        logging.error("No test cases processed. Check your dataset/split.")
        raise ValueError("Expected at leas one test cases")
    
    mean_metrics = metrics_sum / num_cases

    mean_dice, mean_IoU, mean_recall, mean_precision, mean_f1_score = mean_metrics
        
    logging.info(
        "Testing performance : mean_dice %.4f mean_IoU %.4f mean_recall %.4f mean_precision %.4f mean_f1_score %.4f",
        mean_dice, mean_IoU, mean_recall, mean_precision, mean_f1_score
    )

    return "Testing Finished!"





if __name__ == "__main__":
    main()
    


