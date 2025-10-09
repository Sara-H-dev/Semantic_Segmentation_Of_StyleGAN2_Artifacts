import os
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import imageio
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.dataset import SegArtifact_dataset, RandomGenerator
from tqdm import tqdm

def validation_loss(model,
                    device,
                    val_loader,
                    dynamic_loss,
                    bool_break = False, # true if you don't want to go through all validation batches, but want to cancel beforehand
                    n_batches = 0, # Only important if bool_break is true. Number of batches to be validated
                    dataset_path = None,
                    list_dir = './lists',
                    img_size = 1024,
                    ):
    
    val_losses = []
    model.eval()
    with torch.inference_mode():
        for i_batch, sampled_batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            if bool_break:
                if i_batch >= n_batches:
                    break
            
            image = sampled_batch["image"].to(device, non_blocking=True)
            label = sampled_batch["label"].to(device, non_blocking=True)

            # dimensions are checked
            assert image.ndim == 4
            assert label.ndim in (3, 4)

            # forward:
            out_logits = model(image)
            loss = dynamic_loss(out_logits, label)

            val_losses.append(loss.item())
    
    model.train()
    return sum(val_losses) / len(val_losses)


def inference(model, 
              logging,
              testloader,
              test_save_path=None, 
              device = None, 
              split = "test", 
              img_size = 1024,
              sig_threshold = 0.5,
              # bool_break = False, # true if you don't want to go through all validation batches, but want to cancel beforehand
              # n_batches = 0, # Only important if bool_break is true. Number of batches to be validated
              # bool_csv = False
              ):
    
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    num_cases = 0

    metrics_sum = np.zeros(7, dtype=np.float64)  # [dice, IoU, recall, precision, f1, soft_dice, soft_IoU]
    with torch.inference_mode():
        for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader)):
            #if bool_break:
            #    if i_batch > n_batches:
            #        break

            image = sampled_batch["image"].to(device, non_blocking=True)
            label = sampled_batch["label"].to(device, non_blocking=True)
            case_name =  sampled_batch['case_name'][0]

            metric_i = predict_and_score(image, 
                                        label, 
                                        model,  
                                        patch_size = [img_size, img_size],
                                        test_save_path = test_save_path, 
                                        case = case_name, 
                                        device = device,
                                        threshold = sig_threshold)
            
            # Transfer to a robust 1D vector and limit to 7 key figures
            metric_i = np.asarray(metric_i, dtype=np.float64).reshape(-1)[:7]
            
            if metric_i.shape[0] != 7:
                msg = f"Expected 7 metrics, got {metric_i.shape[0]} for case {case_name}"
                logging.error(msg)
                raise ValueError(msg)

            metrics_sum += metric_i
            num_cases += 1
            m_dice, m_iou, m_rec, m_prec, m_f1, m_soft_dice, m_soft_iou = metric_i

            logging.info(
                f"idx {i_batch} case {case_name} "
                f"mean_dice {m_dice:.4f} mean_IoU {m_iou:.4f} "
                f"mean_recall {m_rec:.4f} mean_precision {m_prec:.4f} mean_f1_score {m_f1:.4f} "
                f"mean_soft_dice {m_soft_dice:.4f} mean_soft_IoU {m_soft_iou:.4f} "
            )
    if num_cases == 0:
        logging.error(f"No {split} cases processed. Check your dataset/split.")
        raise ValueError(f"Expected at least one {split} cases")
    
    mean_metrics = metrics_sum / num_cases
    mean_dice, mean_IoU, mean_recall, mean_precision, mean_f1_score, mean_soft_dice, mean_soft_IoU = mean_metrics
        
    logging.info(
        f"{split} performance : mean_dice %.4f mean_IoU %.4f mean_recall %.4f mean_precision %.4f mean_f1_score %.4f mean_soft_dice %.4f mean_soft_IoU %.4f",
        mean_dice, mean_IoU, mean_recall, mean_precision, mean_f1_score, mean_soft_dice, mean_soft_IoU)
    return mean_metrics

def predict_and_score(
        image, 
        label, 
        model, 
        patch_size=[1024, 1024], 
        test_save_path=None, 
        case=None, 
        device = None,
        threshold = 0.5):
    """
    Performs inference on a single 2D image, calculates metrics, and optionally saves the result.
    """
    
    # dimensions are checked
    assert image.ndim == 4
    assert label.ndim in (3, 4)
    # Only batch_size 1 is allowed
    assert image.shape[0] == 1 

    model.eval()

    B, C, H, W = image.shape
    
    image = image.to(device).float()
    if label.ndim == 4:
        label = label.squeeze(1)
    label = label.squeeze(0).to(device).long()

    need_resize = (H, W) != tuple(patch_size)
    # no resize should be necessary.
    assert need_resize == False


    # forward:
    with torch.no_grad():
        out_logits = model(image)
        if out_logits.shape[1] != 1:
            raise ValueError(f"Binary task expected 1 logit channel, got {out_logits.shape[1]}")
        pred = torch.sigmoid(out_logits) 
        pred_bin = (pred > threshold).long().squeeze(0).squeeze(0) #(H, W)

    pred_bin_np = pred_bin.cpu().numpy().astype(np.uint8)
    lab_np  = label.cpu().numpy().astype(np.uint8)

    dice_b, iou_b, recall_b, precision_b, f1_b = calculate_metrics_binary(pred_bin_np, lab_np)
    soft_dice, soft_iou = calculate_metrics_soft(pred.squeeze(0).squeeze(0), label)

    """
    if test_save_path is not None and case is not None:
        os.makedirs(test_save_path, exist_ok=True)
        # Skaliere 0/1-Masken auf 0/255
        imageio.imwrite(os.path.join(test_save_path, f"{case}_pred.png"), pred_bin_np * 255)
        imageio.imwrite(os.path.join(test_save_path, f"{case}_gt.png"),   lab_np * 255)
    """

    return [(dice_b, iou_b, recall_b, precision_b, f1_b, soft_dice, soft_iou)]

def calculate_metrics_binary(pred, gt):
    # makes sure the arrays are binary
    pred = (pred > 0)
    gt   = (gt   > 0)
    smooth = 1e-6

    if pred.sum() > 0 and gt.sum()>0:
        # Dice-Koeffizient
        dice = metric.binary.dc(pred, gt)
        # recall
        recall = metric.binary.recall(pred, gt)
        # precision
        precision = metric.binary.precision(pred, gt)
        # intersection over union (IoU)
        IoU = metric.binary.jc(pred, gt)
        # F1-score
        f1 = 2 * (precision * recall) / (precision + recall + smooth)

        return dice, IoU, recall, precision, f1

    return 0.0, 0.0, 0.0, 0.0, 0.0

def calculate_metrics_soft(pred_map: torch.Tensor, gt: torch.Tensor):
    """
    Soft-Dice und Soft-IoU (Jaccard) mit 'weichen' Zählungen.
    pred_map: (H,W), in [0,1]
    gt:       (H,W), {0,1} oder {0,255}
    """
    eps = 1e-6
    pred = pred_map.float().view(-1)
    gtruth = gt.float().view(-1)
    if gtruth.max() > 1:  # normalisiere {0,255} -> {0,1}
        gtruth = gtruth / 255.0

    intersection = (pred * gtruth).sum()
    sum_p_2 = (pred * pred).sum()
    sum_g_2 = (gtruth * gtruth).sum()
    sum_p = pred.sum()
    sum_g = gtruth.sum()

    # Soft Dice 
    soft_dice = (2.0 * intersection + eps) / (sum_p_2 + sum_g_2 + eps)

    # Soft IoU (Jaccard)
    soft_iou = (intersection + eps) / (sum_p + sum_g - intersection + eps)

    return soft_dice.item(), soft_iou.item()
