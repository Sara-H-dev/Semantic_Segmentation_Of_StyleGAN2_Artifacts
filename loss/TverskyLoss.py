import torch
import torch.nn as nn

class TverskyLoss_binary(nn.Module):

    def __init__(
            self,
            alpha: float = 0.4,
            beta: float = 0.6,):
        super(TverskyLoss_binary, self).__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, y_pred, y_gtruth):
        # y_pred and y_gtruth have the shape B, 1, H, W -> binary classification
        smooth = 1e-6   # smooth is needed
                        # if no artefact are in the image
                        # so TP = 0 and FN = 0
                        # but FP gets ignort

        if y_gtruth.dim() == 3:  # (B,H,W) -> (B,1,H,W)
            y_gtruth = y_gtruth.unsqueeze(1)     

        y_gtruth = y_gtruth.float()

        # normalize labels if {0,255}
        if y_gtruth.max() > 1:
            y_gtruth = y_gtruth / 255.0  


        # Logits -> propability
        y_pred = torch.sigmoid(y_pred)

        # Flatten per Sample
        # to dimension (B, N)
        # B = Batchsize, N = number of pixel per picture
        y_pred   = y_pred.view(y_pred.size(0), -1)
        y_gtruth = y_gtruth.view(y_gtruth.size(0), -1)

        # .sum(dim=1) sums over the pixel dimension N.
        # output dim = B, one value per image
        TP = (y_gtruth * y_pred).sum(dim=1)
        FP = ((1.0 - y_gtruth) * y_pred).sum(dim=1)
        FN = (y_gtruth * (1.0 - y_pred)).sum(dim=1)

        tversky = 1.0 - ((TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth))
        # mean other all pictures
        # = one value
        return tversky.mean()
    
    