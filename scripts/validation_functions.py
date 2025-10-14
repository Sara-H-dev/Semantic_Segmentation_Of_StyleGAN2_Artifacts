import numpy as np
import torch
from medpy import metric
from tqdm import tqdm
import math

# -------------------- Validation Loss ---------------------------- #
def validation_loss(model,
                    device,
                    val_loader,
                    dynamic_loss,
                    bool_break = False, # true if you don't want to go through all validation batches, but want to cancel beforehand
                    n_batches = 0): # Only important if bool_break is true. Number of batches to be validate   
    val_losses = []
    model.eval()
    with torch.inference_mode():
        for i_batch, sampled_batch in enumerate(val_loader):
            # breaks the validation calculation after n_batches
            if bool_break and (i_batch >= n_batches): break
                
            image = sampled_batch["image"].to(device, non_blocking=True)
            label = sampled_batch["label"].to(device, non_blocking=True)

            # dimensions are checked
            assert image.ndim == 4; assert label.ndim in (3, 4)
            # forward:
            out_logits = model(image)
            loss = dynamic_loss(out_logits, label)
            val_losses.append(loss.item())
    model.train()
    mean_val_loss = sum(val_losses) / len(val_losses)
    return mean_val_loss

# -------------------- Metrics ----------------------------------------------- #
def calculate_metrics(  model, 
                        logging,
                        testloader,
                        dynamic_loss,
                        csv_epoch,
                        csv_batch,
                        csv_confusion_epoch,
                        csv_confusion_batch,
                        epoch,
                        device = None, 
                        split = "test", 
                        img_size = 1024,
                        sig_threshold = 0.5,):
    patch_size = (img_size, img_size)
    model.eval(); 
    num_cases = 0;  # number of validation runs
    ten_output_saver = [] # list to save ten outputs for generating heat maps for the best run
    metric_list = []
    confusion_list = []

    with torch.inference_mode():
        # ------------- calculate the metric and predict ---------------------------------------------
        for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader)):
            image = sampled_batch["image"].to(device, non_blocking=True)
            label = sampled_batch["label"].to(device, non_blocking=True)
            case_name =  sampled_batch['case_name'][0]
            skip_flag = False # for real image the metrics can not be calculated, so they are scippt

            # --------- redimension images and labels ------------------
            assert image.ndim == 4; assert label.ndim in (3, 4); assert image.shape[0] == 1 
            B, C, H, W = image.shape
            image = image.to(device).float()
            if label.ndim == 4: label = label.squeeze(1)
            label = label.squeeze(0).to(device).long()
            need_resize = (H, W) != tuple(patch_size); assert need_resize == False  # no resize should be necessary.
            
            # =========== forward: ==============================================================
            with torch.no_grad():
                out_logits = model(image)
                val_loss = dynamic_loss(out_logits, label)
                if out_logits.shape[1] != 1:
                    raise ValueError(f"Binary task expected 1 logit channel, got {out_logits.shape[1]}")
                pred = torch.sigmoid(out_logits) 
                pred_bin = (pred > sig_threshold).long().squeeze(0).squeeze(0) #(H, W)

            pred_bin_np = pred_bin.cpu().numpy().astype(np.uint8)
            lab_np  = label.cpu().numpy().astype(np.uint8)

            # ===================== calculating the metrics ======================================
            i_dice, i_iou, i_rec, i_prec, i_f1 = calculate_metrics_binary(pred_bin_np, lab_np)
            i_soft_dice, i_soft_iou, true_pos, false_pos, false_neg, true_neg = calculate_metrics_soft(pred.squeeze(0).squeeze(0), label)
            if any(math.isnan(x) for x in [i_dice, i_iou, i_rec, i_prec, i_f1]):
                skip_flag = True

            if skip_flag == False:
                metric_i = np.array([
                    i_dice, i_iou, i_rec, i_prec, i_f1, i_soft_dice, i_soft_iou, val_loss.item(),
                    true_pos, false_pos, false_neg, true_neg ], dtype=np.float64)
                metric_list.append(torch.tensor(metric_i, dtype=torch.float64))
                csv_batch.writerow([epoch, i_batch, case_name,  i_dice, i_iou, i_rec, i_prec, i_f1, i_soft_dice,i_soft_iou, val_loss.item(), true_pos, false_pos, false_neg, true_neg])
            
            # ------------------ out tupel ---------------
            out_tuple = (case_name, image, pred) # tupel for ploting the heat map of the best run
            if(i_batch < 10): 
                ten_output_saver.append(out_tuple) # list to save ten outputs for generating heat maps for the best run
            num_cases += 1

    if num_cases == 0:
        logging.error(f"No {split} cases processed. Check your dataset/split.")
        raise ValueError(f"Expected at least one {split} cases")
    
    # -------- calculating the average --------------------------
    valid_cases = len(metric_list)
    if valid_cases == 0:
        raise ValueError(f"No valid {split} metrics to aggregate.")
    metrics_all = torch.stack(metric_list, dim=0)
    mean_metrics =  torch.nanmean(metrics_all, dim=0) # can ignor nans
    mean_vals = mean_metrics.tolist()
    (mean_dice, mean_IoU, mean_recall, mean_precision, mean_f1_score,
    mean_soft_dice, mean_soft_IoU, mean_val_loss,
    mean_true_pos, mean_false_pos, mean_false_neg, mean_true_neg) = mean_vals
    
    csv_epoch.writerow([epoch, *mean_vals])
    csv_confusion_epoch.writerow(epoch, true_pos, true_neg, false_pos, false_neg)

    logging.info(
        f"{split} performance : mean_dice {mean_dice} mean_IoU {mean_IoU} mean_recall {mean_recall} mean_precision {mean_precision}"
        f"mean_f1_score {mean_f1_score} mean_soft_dice {mean_soft_dice} mean_soft_IoU {mean_soft_IoU} mean_val_loss {mean_val_loss}"
        f"mean_true_pos {mean_true_pos}  mean_false_pos {mean_false_pos}  mean_false_neg {mean_false_neg}  mean_true_neg {mean_true_neg} ")

    return mean_dice, ten_output_saver


# -------------------- Calculating BINARY Metrics ---------------------------- #
def calculate_metrics_binary(pred, ground_truth):
    # makes sure the arrays are binary
    pred = (pred > 0)
    ground_truth   = (ground_truth   > 0)
    smooth = 1e-6

    if pred.sum() == 0 and ground_truth.sum() == 0:
        return 1, 1, 1, 1, 1
    
    elif ground_truth.sum() == 0 and ground_truth.sum() > 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    
    elif ground_truth.sum() > 0 and ground_truth.sum() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    # Dice-Koeffizient
    dice = metric.binary.dc(pred, ground_truth)
    # recall
    recall = metric.binary.recall(pred, ground_truth)
    # precision
    precision = metric.binary.precision(pred, ground_truth)
    # intersection over union (IoU)
    IoU = metric.binary.jc(pred, ground_truth)
    # F1-score
    f1 = 2 * (precision * recall) / (precision + recall + smooth)

    return dice, IoU, recall, precision, f1

# -------------------- Calculating SOFT Metrics ---------------------------- #
def calculate_metrics_soft(pred_map, ground_truth):
    """
    pred_map: (H,W), in [0,1]
    ground_truth:       (H,W), {0,1} or {0,255}
    """
    eps = 1e-8
    pred = pred_map.float().view(-1)
    gtruth = ground_truth.float().view(-1)

    if gtruth.max() > 1:  # normalisiere {0,255} -> {0,1}
        gtruth = gtruth / 255.0

    intersection = torch.sum(pred * gtruth)
    sum_p_2 = torch.sum(pred * pred)
    sum_g_2 = torch.sum(gtruth * gtruth)
    sum_p = torch.sum(pred)
    sum_g = torch.sum(gtruth)

    true_pos = torch.sum(pred * gtruth)
    false_pos = torch.sum((1 - gtruth) * pred)
    false_neg = torch.sum(gtruth * (1 - pred))
    true_neg = torch.sum((1 - pred) * (1 - gtruth))

    accuracy = 
    
    # Soft Dice 
    i_soft_dice = (2.0 * intersection + eps) / (sum_p_2 + sum_g_2 + eps)

    # Soft IoU (Jaccard)
    i_soft_iou = (intersection + eps) / (sum_p + sum_g - intersection + eps)

    return i_soft_dice.item(), i_soft_iou.item(), true_pos.item(), false_pos.item(), false_neg.item(), true_neg.item()
