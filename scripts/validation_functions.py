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
    if len(val_losses) <= 0:
        return float("nan")
    mean_val_loss = sum(val_losses) / len(val_losses)
    return mean_val_loss

# -------------------- Metrics ----------------------------------------------- #
def calculate_metrics(  model, 
                        logging,
                        testloader,
                        dynamic_loss,
                        csv_all_epoch,
                        csv_fake_epoch,
                        csv_real_epoch,
                        csv_batch_real,
                        csv_batch_fake,
                        mean_train_loss,
                        epoch,
                        device = None, 
                        split = "test", 
                        img_size = 1024,
                        sig_threshold = 0.5,
                        output_num = 10):
    
    if output_num >= len(testloader):
        output_num = len(testloader)
    
    patch_size = (img_size, img_size)
    model.eval(); 
    num_cases = 0;  # number of validation runs
    ten_output_saver = [] # list to save ten outputs for generating heat maps for the best run
    
    # real
    real_conf_matrix_bin_list = []  # list of [[tp, fp],[fn, tn]]
    real_confusion_matrix_soft_list = []
    accuracy_list_real = []         # [(acc, val_loss), ...]
    real_image_counter = 0
    FRP_list = []
    # fake
    confusion_matrix_soft_list = [] # list of [[tp, fp],[fn, tn]]
    fake_conf_matrix_bin_list = []  # list of [[tp, fp],[fn, tn]]
    fake_confusion_matrix_soft_list = []
    accuracy_list_fake = []         # [(acc, val_loss), ...]
    metric_fake_list = []           # [[acc, rec, prec, val_loss, IoU, dice, f1, soft_dice, ..], ..]
    # all
    accuracy_list = []              # [(acc, val_loss), ...]
    confusion_matrix_bin_list = []  # list of [[tp, fp],[fn, tn]]

    with torch.inference_mode():
        # ------------- calculate the metric and predict ---------------------------------------------
        for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader)):

            image = sampled_batch["image"].to(device, non_blocking=True)
            loss_label = sampled_batch["label"].to(device, non_blocking=True)
            case_name =  sampled_batch['case_name'][0]

            # --------- redimension images and labels ------------------
            assert image.ndim == 4
            assert loss_label.ndim in (3, 4)
            assert image.shape[0] == 1 

            B, C, H, W = image.shape
            assert ((H, W) != tuple(patch_size)) == False 
            image = image.to(device).float()

            if loss_label.ndim == 4: 
                label = loss_label.squeeze(1) # B, H, W
            else: label = loss_label

            # =========== forward: ==============================================================
            out_logits = model(image)
            if out_logits.shape[1] != 1:
                raise ValueError(f"Binary task expected 1 logit channel, got {out_logits.shape[1]}")
            val_loss = dynamic_loss(out_logits, loss_label)

            # probabilities & binaries for metrics
            pred = torch.sigmoid(out_logits).squeeze(0).squeeze(0)        # (H,W) float
            pred_bin = (pred > sig_threshold)                             # (H,W) bin
            ground_truth = (label.squeeze(0) > 0)                  # (H,W) bin

            # ===================== calculating the metrics ======================================
            #real picture
            if not ground_truth.any().item(): 
                real_image_counter += 1
                confusion_matrix_bin, confusion_matrix_soft, accuracy, FRP = calculate_metrics_real(pred_bin, pred, ground_truth)
                #csv_batch_real.writerow([epoch, i_batch, accuracy, confusion_matrix_bin, float(val_loss)]) 

                confusion_matrix_bin_list.append(confusion_matrix_bin)
                real_conf_matrix_bin_list.append(confusion_matrix_bin)
                real_confusion_matrix_soft_list.append(confusion_matrix_soft)
                confusion_matrix_soft_list.append(confusion_matrix_soft)
                accuracy_list.append((accuracy, float(val_loss)))
                accuracy_list_real.append((accuracy, float(val_loss)))
                FRP_list.append(float(FRP))
            # fake picture
            else:
                (bin_accuracy, bin_recall,bin_precision,
                    bin_IoU, bin_dice, bin_f1,
                    confusion_matrix_bin, confusion_matrix_soft,
                    i_soft_dice, i_soft_iou) = calculate_metrics_fake(pred_bin, pred, ground_truth)
                #csv_batch_fake.writerow([epoch, i_batch, bin_accuracy, bin_recall, bin_precision, float(val_loss), bin_IoU, bin_dice, bin_f1, confusion_matrix_bin, confusion_matrix_soft, i_soft_dice, i_soft_iou])

                metric_fake_list.append([bin_accuracy, bin_recall, bin_precision, bin_IoU, bin_dice, bin_f1, i_soft_dice, i_soft_iou])
                confusion_matrix_bin_list.append(confusion_matrix_bin)
                fake_conf_matrix_bin_list.append(confusion_matrix_bin)
                confusion_matrix_soft_list.append(confusion_matrix_soft)
                fake_confusion_matrix_soft_list.append(confusion_matrix_soft)
                accuracy_list.append((bin_accuracy, float(val_loss)))
                accuracy_list_fake.append((bin_accuracy, float(val_loss)))
            
            # ------------------ out tupel ---------------
            out_tuple = (case_name, image, pred) # tupel for ploting the heat map of the best run
            if(i_batch < output_num): 
                ten_output_saver.append(out_tuple) # list to save ten outputs for generating heat maps for the best run
            num_cases += 1

    if num_cases == 0:
        logging.error(f"No {split} cases processed. Check your dataset/split.")
        raise ValueError(f"Expected at least one {split} cases")
    
    # -------- calculating the average --------------------------
    valid_fake_cases = len(metric_fake_list)
    if valid_fake_cases == 0:
        raise ValueError(f"No valid fake {split} metrics to aggregate.")
    
    # real images mean metrics
    if real_image_counter > 0:
        mean_acc_and_loss =  np.mean(np.array(accuracy_list_real, dtype=float), axis=0)
        mean_confusion_matrix_bin_real = np.mean(np.array(real_conf_matrix_bin_list, dtype=float), axis=0).flatten().tolist()
        mean_confusion_matrix_soft_real = np.mean(np.array(real_confusion_matrix_soft_list, dtype=float), axis=0).flatten().tolist()
        mean_FPR = np.mean(np.array(FRP_list, dtype=float), axis=0)

        (mean_accuracy_real, mean_val_loss_real) = mean_acc_and_loss
        csv_real_epoch.writerow([epoch, float(mean_accuracy_real), mean_confusion_matrix_bin_real, mean_confusion_matrix_soft_real, mean_val_loss_real, mean_FPR])
        logging.info(f"{split} real performance for epoch {epoch} :"
                     f" mean_confusion_matrix_bin [[tp, fp],[fn, tn]] {mean_confusion_matrix_bin_real} "
                     f" mean_accuracy {mean_accuracy_real} mean_val_loss{mean_val_loss_real}")
    # fake images mean metrics

    (mean_accuracy_fake, mean_val_loss_fake) = np.mean(np.array(accuracy_list_fake, dtype=float), axis=0)

    mean_confusion_matrix_bin_fake = np.mean(np.array(fake_conf_matrix_bin_list, dtype=float), axis=0).flatten().tolist()
    mean_confusion_matrix_soft_fake = np.mean(np.array(fake_confusion_matrix_soft_list, dtype=float), axis=0).flatten().tolist()
    
    
    mean_fake_metric = np.mean(np.array(metric_fake_list, dtype=float), axis=0)
    (mean_bin_accuracy, mean_bin_recall, mean_bin_precision, mean_bin_IoU, mean_bin_dice, mean_bin_f1, mean_soft_dice, mean_soft_iou) = mean_fake_metric

    Score = mean_soft_dice - (10 * mean_FPR)
    
    csv_fake_epoch.writerow([epoch, float(mean_accuracy_fake), 
                float(mean_val_loss_fake), mean_confusion_matrix_bin_fake, 
                mean_confusion_matrix_soft_fake,  *[float(x) for x in mean_fake_metric]])  
    logging.info(
        f"{epoch}_fake: mean_soft_dice {mean_soft_dice} mean_val_loss {mean_val_loss_fake} mean_bin_recall {mean_bin_recall} mean_bin_precision {mean_bin_precision} mean_bin_dice {mean_bin_dice}"
)

    # accuracy confusion matrix and val loss for all images
    (mean_accuracy, mean_val_loss) = np.mean(np.array(accuracy_list, dtype=float), axis=0)
    mean_confusion_matrix_bin = np.mean(np.array(confusion_matrix_bin_list, dtype=float), axis=0).flatten().tolist()
    mean_confusion_matrix_soft = np.mean(np.array(confusion_matrix_soft_list, dtype=float), axis=0).flatten().tolist()

    csv_all_epoch.writerow([
        epoch,
        float(mean_accuracy),
        float(mean_val_loss),
        float(mean_train_loss),
        mean_confusion_matrix_bin,
        mean_confusion_matrix_soft,
        Score
    ])

   
    logging.info(f"{split} epoch {epoch}: mean_accuracy {mean_accuracy} "
                 f"mean_cofusion_matrix [[tp, fp],[fn, tn]]{mean_confusion_matrix_bin} "
                 f"mean_val_loss {mean_val_loss}")
    
    print(f"epoch{epoch} val_loss:{mean_val_loss} train_loss:{mean_train_loss} mean_soft_dice:{mean_soft_dice} mean_FRP {mean_FPR} Score {Score}")

    return mean_soft_dice, ten_output_saver, Score, mean_FPR

# -------------------- Calculating REAL Metrics ---------------------------- #
def calculate_metrics_real(pred_bin, pred, ground_truth):
    pred_bin = pred_bin.bool()
    ground_truth = ground_truth.bool()

    # Binary Calculation
    tp = torch.sum(pred_bin & ground_truth).item()
    fp = torch.sum(pred_bin & (~ground_truth)).item()
    fn = torch.sum((~pred_bin) & ground_truth).item()
    tn = torch.sum((~pred_bin) & (~ground_truth)).item()

    false_pos = torch.sum((~ground_truth).float() * pred)
    false_neg = torch.sum(ground_truth.float() * (1.0 - pred))
    true_pos  = torch.sum(ground_truth.float() * pred)
    true_neg  = torch.sum((~ground_truth).float() * (1.0 - pred))

    # confusion matrix
    confusion_matrix_soft = [
        [float(true_pos.item()), float(false_pos.item())],
        [float(false_neg.item()), float(true_neg.item())]
    ]

    FPR = fp / (fp + tn)

    # accuracy
    total = tp + tn + fp + fn
    if total <= 0: raise ValueError(f"Real metric calculation failed because total = {total}")  
    accuracy = (tp + tn) / total

    confusion_matrix_bin = [[tp, fp],[fn, tn]]

    return confusion_matrix_bin, confusion_matrix_soft, float(accuracy), FPR

# -------------------- Calculating FAKEs Metrics ---------------------------- #
def calculate_metrics_fake(pred_bin, pred, ground_truth):
    smooth = 1e-8
    ground_truth = ground_truth.bool()
    pred_bin = pred_bin.bool()
    gt_np   = ground_truth.cpu().numpy()
    predb_np = pred_bin.cpu().numpy()

    # =========== BINARY CALCULATION===================

    # Dice-Koeffizient
    bin_dice = metric.binary.dc(predb_np, gt_np)
    # recall
    bin_recall = metric.binary.recall(predb_np, gt_np)
    # precision
    bin_precision = metric.binary.precision(predb_np, gt_np)
    # intersection over union (IoU)
    bin_IoU = metric.binary.jc(predb_np, gt_np)
    # F1-score
    bin_f1 = 2 * (bin_precision * bin_recall) / (bin_precision + bin_recall + smooth)

    tp = torch.sum(pred_bin & ground_truth).item()
    fp = torch.sum(pred_bin & (~ground_truth)).item()
    fn = torch.sum((~pred_bin) & ground_truth).item()
    tn = torch.sum((~pred_bin) & (~ground_truth)).item()

    confusion_matrix_bin = [[tp, fp],[fn, tn]]

    # accuracy
    total = tp + tn + fp + fn
    if total <= 0: raise ValueError(f"Real metric calculation failed because total = {total}")  
    bin_accuracy = (tp + tn) / total

    # =========== SOFT CALCULATION===================
    pred = pred.float().view(-1)
    gtruth = ground_truth.float().view(-1)

    intersection = torch.sum(pred * gtruth)
    sum_p_2 = torch.sum(pred * pred)
    sum_g_2 = torch.sum(gtruth * gtruth)
    sum_p = torch.sum(pred)
    sum_g = torch.sum(gtruth)

    true_pos = torch.sum(pred * gtruth)
    false_pos = torch.sum((1 - gtruth) * pred)
    false_neg = torch.sum(gtruth * (1 - pred))
    true_neg = torch.sum((1 - pred) * (1 - gtruth))

    # confusion matrix
    confusion_matrix_soft = [
        [float(true_pos.item()), float(false_pos.item())],
        [float(false_neg.item()), float(true_neg.item())]
    ]

    # Soft Dice 
    i_soft_dice = float((2.0 * intersection + smooth) / (sum_p_2 + sum_g_2 + smooth))

    # Soft IoU (Jaccard)
    i_soft_iou = float((intersection + smooth) / (sum_p + sum_g - intersection + smooth))

    return (float(bin_accuracy), float(bin_recall), float(bin_precision),
            float(bin_IoU), float(bin_dice), float(bin_f1),
            confusion_matrix_bin, confusion_matrix_soft,
            i_soft_dice, i_soft_iou)
