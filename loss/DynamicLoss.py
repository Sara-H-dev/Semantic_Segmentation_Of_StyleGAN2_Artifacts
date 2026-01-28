import torch
import torch.nn.functional as F
import torch.nn as nn


class BCEWithLogitsLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        loss = self.bce(outputs, targets)
        a = 0

        if torch.isnan(loss):
            if not torch.isfinite(loss):
                a = 1
            else:
                raise ValueError("BCE loss is nan but not infinite")
        return loss
    
class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)
        targets = targets.float()
        dims_i = inputs.ndim
        dims_t = targets.ndim

        if (dims_i != dims_t):
            raise ValueError(f"target dim {dims_t} is not equal to input dim {dims_i}")
        dims = dims_i

        if dims == 3:
            inputs = inputs.squeeze(0) # H, W
            targets = targets.squeeze(0) # H, W
        dims = inputs.ndim

        true_pos = (inputs * targets).sum(dim=(0,1))
        false_pos = (inputs * (1 - targets)).sum(dim=(0,1))
        false_neg = ((1 - inputs) * targets).sum(dim=(0,1))

        tversky_index = (true_pos + smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + smooth)
        loss = 1 - tversky_index

        if loss > 1:
            return ValueError(f"Tversky Loss is higher than 1, it is {loss}")
        return loss
    
class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.1):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)
        true_pos = torch.sum(inputs * targets)
        false_pos = torch.sum((1 - targets) * inputs)
        false_neg = torch.sum(targets * (1 - inputs))

        tversky_index = (true_pos + smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + smooth
        )
        return (1 - tversky_index) ** self.gamma

#Dynamic Loss Function
class DynamicLoss(torch.nn.Module):
    def __init__(self, roi_thresh=0.04, alpha = 0.4, beta = 0.6, tversky_bce_mix = 0.5):
        super(DynamicLoss, self).__init__()
        self.roi_thresh = roi_thresh
        self.bce_loss = BCEWithLogitsLoss()
        self.tversky_loss = TverskyLoss(alpha = alpha, beta = beta)
        self.focal_tversky_loss = FocalTverskyLoss(alpha = alpha, beta = beta)
        self.tversky_bce_mix = tversky_bce_mix

    def forward(self, output, target):

        if target.dim() == 3:  # (B,H,W) -> (B,1,H,W)
            target = target.unsqueeze(1)
        target = target.float()
        if target.max() > 1:
            target = (target > 127.5).float()

        B = output.size(0)
        losses = []
        B_t = target.size(0)
        if B != B_t:
            raise ValueError(f"Batchsize from ouptut {B} not equal to batchsize target {B_t}")
        
        for i in range(B):
            output_i = output[i]
            target_i = target[i]

            #loss_i = self.bce_loss(output_i, target_i)
            loss_bce_i = self.bce_loss(output_i, target_i)
            loss_tversky_i = 0
            if torch.sum(target_i) != 0: 
                loss_tversky_i = self.tversky_loss(output_i, target_i)

                loss_i = (1-self.tversky_bce_mix) * loss_bce_i + self.tversky_bce_mix * loss_tversky_i 
            else: loss_i = loss_bce_i

            losses.append(loss_i)

        return torch.mean(torch.stack(losses))
        