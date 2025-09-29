import os
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import imageio

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
        f1 = 2 * (precision * recall) / (precision + recall)

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