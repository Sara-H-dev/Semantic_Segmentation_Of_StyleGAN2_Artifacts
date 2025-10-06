import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################
#  Symmetric Unified Focal Loss (1-Kanal Logits)
##############################################
class SymmetricUnifiedFocalLoss(nn.Module):
    """
    Combines Symmetric Focal Loss and Symmetric Focal Tversky Loss
    for single-channel logits (binary segmentation).
    """

    def __init__(self, weight: float = 0.5, delta: float = 0.6, gamma: float = 0.5):
        super().__init__()
        self.weight = weight
        self.floss = SymmetricFocalLoss(delta, gamma)
        self.ftloss = SymmetricFocalTverskyLoss(delta, gamma)

    def forward(self, y_pred, y_true):
        picture_is_real = False
        if torch.all(y_true == 0):
            picture_is_real = True


        if y_true.dim() == 3:  # (B,H,W) -> (B,1,H,W)
            y_true = y_true.unsqueeze(1)     

        y_true = y_true.float()

        # normalize labels if {0,255}
        if y_true.max() > 1:
            y_true = y_true / 255.0 

        loss_f  = self.floss(y_pred, y_true)
        loss_ft = self.ftloss(y_pred, y_true)

        if picture_is_real:
            return loss_f
        else:
            if self.weight is not None:
                return self.weight * loss_ft + (1 - self.weight) * loss_f
            else:
                return loss_ft + loss_f


##############################################
#     Symmetric Focal Loss (1-Kanal Logits)  #
##############################################
class SymmetricFocalLoss(nn.Module):
    """
    Binary symmetric focal loss for logits.
    y_pred: raw logits (B,1,H,W)
    y_true: binary mask (B,1,H,W) or (B,H,W)
    """

    def __init__(self, delta: float = 0.7, gamma: float = 2.0):
        super().__init__()
        self.delta = delta
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        eps = 1e-7

        # BCE ohne Reduktion
        bce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")

        # Sigmoid für p_t-Berechnung
        p = torch.sigmoid(y_pred)
        pt = p * y_true + (1 - p) * (1 - y_true)

        # Klassen-Gewichtung (delta = FG-Gewicht)
        w = self.delta * y_true + (1 - self.delta) * (1 - y_true)

        # Symmetric Focal Loss
        loss = w * (1 - pt).pow(1 - self.gamma) * bce
        return loss.mean()


#################################################
#  Symmetric Focal Tversky Loss (1-Kanal Logits)
#################################################
class SymmetricFocalTverskyLoss(nn.Module):
    """
    Binary symmetric focal Tversky loss for logits.
    y_pred: raw logits (B,1,H,W)
    y_true: binary mask (B,1,H,W) or (B,H,W)
    """

    def __init__(self, delta: float = 0.7, gamma: float = 0.75):
        super().__init__()
        self.delta = delta
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        eps = 1e-7

        # Logits → Wahrscheinlichkeiten
        y_prob = torch.sigmoid(y_pred)

        # True/False Positives/Negatives
        y_prob   = y_prob.view(y_prob.size(0), -1)
        y_true = y_true.view(y_true.size(0), -1)
        
        tp = (y_true * y_prob).sum(dim=1)
        fn = (y_true * (1 - y_prob)).sum(dim=1)
        fp = ((1 - y_true) * y_prob).sum(dim=1)

        ti = (tp + eps) / (tp + self.delta * fn + (1 - self.delta) * fp + eps)

        loss = (1 - ti).pow(self.gamma)
        return loss.mean()

