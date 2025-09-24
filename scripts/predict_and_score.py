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
        out = torch.sigmoid(out_logits) 
        pred_bin = (out > threshold).long().squeeze(0).squeeze(0) #(H, W)

    pred_np = pred_bin.cpu().numpy().astype(np.uint8)
    lab_np  = label.cpu().numpy().astype(np.uint8)

    dice, iou, recall, precision, f1 = calculate_metrics_binary(pred_np, lab_np)

    if test_save_path is not None and case is not None:
        os.makedirs(test_save_path, exist_ok=True)
        # Skaliere 0/1-Masken auf 0/255
        imageio.imwrite(os.path.join(test_save_path, f"{case}_pred.png"), pred_np * 255)
        imageio.imwrite(os.path.join(test_save_path, f"{case}_gt.png"),   lab_np * 255)

    return [(dice, iou, recall, precision, f1)]

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