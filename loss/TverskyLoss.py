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

        torch.flatten(y_pred)
        torch.flatten(y_gtruth)

        TP = torch.sum(y_gtruth * y_pred)
        FP = torch.sum((1 - y_gtruth) * y_pred)
        FN = torch.sum(y_gtruth * (1 - y_pred))

        tversky = 1 - (TP + smooth) / (TP + self.alpha * FP, self.beta * FN + self.smooth)
        return tversky
    
    