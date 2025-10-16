import torch
import torch.nn as nn
import torch.nn.functional as F

class SYM_UIFIED_FOCAL_LOSS(torch.nn.Module):
    def __init__(self, weight=0.5, delta=0.6, gamma=0.5):
        super().__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma

    def forward(self, pred, g_truth):
        if g_truth.dim() == 3:  # (B,H,W) -> (B,1,H,W)
            g_truth = g_truth.unsqueeze(1)
        g_truth = g_truth.float()
        if g_truth.max() > 1:
            g_truth = (g_truth > 127.5).float()

        B = pred.size(0)
        B_t = g_truth.size(0)
        if B != B_t:
            raise ValueError(f"Batchsize from ouptut {B} not equal to batchsize g_truth {B_t}")
        losses_ftl = []
        losses_fl = []
        
        for i in range(B):
            y_true = g_truth[i] # -> (1, H, W)
            y_pred = pred[i] # -> (1, H, W)
            y_pred = y_pred.squeeze(0) # -> (H,W)
            y_true = y_true.squeeze(0) # -> (H,W)
            y_pred = torch.sigmoid(y_pred)
            
            losses_ftl.append(self.symmetric_focal_tversky_loss(y_true,y_pred))
            losses_fl.append(self.symmetric_focal_loss(y_true,y_pred))

        symmetric_ftl = torch.stack(losses_ftl).mean()
        symmetric_fl  = torch.stack(losses_fl ).mean()

        if self.weight is not None:
            return (self.weight * symmetric_ftl) + ((1 - self.weight) * symmetric_fl)  
        else:
            return symmetric_ftl + symmetric_fl

    def symmetric_focal_loss(self, y_true,y_pred):
        # y_true -> (H,W)
        # y_pred -> (H,W)

        epsilon = 1e-8
        y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon) # so it is no 0 nor 1 , because this would lead to instability

        cross_entropy_artefakt = -y_true * torch.log(y_pred)
        cross_entropy_background = - (1 - y_true) * torch.log(1 - y_pred)
        #calculate losses separately for each class

        fore_ce = (1 - y_pred) ** self.gamma * cross_entropy_artefakt
        fore_ce = self.delta * fore_ce

        back_ce = (y_pred) ** self.gamma * cross_entropy_background
        back_ce = (1 - self.delta) * back_ce

        loss = (fore_ce + back_ce).mean()

        return loss


    def symmetric_focal_tversky_loss(self, y_true,y_pred):
        # y_true -> (H,W)
        # y_pred -> (H,W)
        # background
        z_true = 1 - y_true
        z_pred = 1 - y_pred

        epsilon = 1e-8
        y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)

        tp = (y_pred * y_true).sum(dim=(0,1))
        fp = (y_pred * (1 - y_true)).sum(dim=(0,1))
        fn = ((1 - y_pred) * y_true).sum(dim=(0,1))
        #tn = ((1 - y_pred) * (1 - y_true)).sum(dim=(0,1))

        TI_for = (tp + epsilon)/(tp + self.delta*fn + (1-self.delta)*fp + epsilon)

        tp = (z_pred * z_true).sum(dim=(0,1))
        fp = (z_pred * (1 - z_true)).sum(dim=(0,1))
        fn = ((1 - z_pred) * z_true).sum(dim=(0,1))
        #tn = ((1 - z_pred) * (1 - z_true)).sum(dim=(0,1))

        TI_back = (tp + epsilon)/(tp + self.delta*fn + (1-self.delta)*fp + epsilon)

        fore_dice = (1.0 - TI_for)  * torch.pow(1.0 - TI_for,  -self.gamma)
        back_dice = (1.0 - TI_back) * torch.pow(1.0 - TI_back, -self.gamma)

        # Average class scores
        loss = (fore_dice + back_dice).mean()
        return loss