import torch
import torch.nn as nn
import torch.nn.functional as F

class SymmetricUnifiedFocalLoss(nn.Module):

    def __init__(
            self,
            weight: float = 0.5,
            delta: float = 0.6,
            gamma: float = 0.5):
        super(SymmetricUnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
    
    def forward(self, logits_pred, y_gtruth):

        real_img = False
        

        if y_gtruth.dim() == 3:  # (B,H,W) -> (B,1,H,W)
            y_gtruth = y_gtruth.unsqueeze(1)     

        y_gtruth = y_gtruth.float()

        # normalize labels if {0,255}
        if y_gtruth.max() > 1:
            y_gtruth = y_gtruth / 255.0 

        if y_gtruth.sum().item() == 0:
            real_img = True


        # Flatten per Sample
        # to dimension (B, N)
        # B = Batchsize, N = number of pixel per picture
        logits_pred   = logits_pred.view(logits_pred.size(0), -1)
        y_gtruth = y_gtruth.view(y_gtruth.size(0), -1)

        # Logits -> propability
        y_pred = torch.sigmoid(logits_pred)

        if real_img == False:
            sym_focal_tversky_loss = symmetric_focal_tversky_loss(delta=self.delta, gamma=self.gamma, y_true = y_gtruth, y_pred = y_pred)
        sym_focal_loss = symmetric_focal_loss(delta=self.delta, gamma=self.gamma, y_true = y_gtruth, pred_logits = logits_pred)

        if self.weight is not None:
            if real_img == True:
                return sym_focal_loss
            else:
                return (self.weight * sym_focal_loss) + ((1 - self.weight) * sym_focal_tversky_loss)  
        else:
            raise ValueError("weight is none")
        
#################################
# Symmetric Focal Tversky loss  #
#################################
def symmetric_focal_tversky_loss(delta, gamma, y_true, y_pred):
 
    # values to prevent division by zero error
    epsilon = 1e-7

    tp = (y_true * y_pred).sum(dim=1)
    fp = ((1.0 - y_true) * y_pred).sum(dim=1) #FP = g1ᵢ·p0ᵢ 
    fn = (y_true * (1.0 - y_pred)).sum(dim=1) #FN = g0ᵢ·p1ᵢ

    mTI = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)

    loss = (1 - mTI).pow(gamma)

    return loss.mean()

################################
#       Symmetric Focal loss      #
################################
def symmetric_focal_loss(delta, gamma, y_true, pred_logits):

    L_bce = F.binary_cross_entropy_with_logits(pred_logits, y_true, reduction='none')

    y_pred = torch.sigmoid(pred_logits)

    pt = y_pred * y_true + (1 - y_pred) * (1 - y_true)

    loss = delta * (1 - pt).pow(1 - gamma) * L_bce

    return loss.mean()
