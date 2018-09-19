from src.imports import *

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, label_pred, label_target, bb_preds, bb_targets):
        batch_size,_ = label_target.size()

        clf_loss = F.binary_cross_entropy(label_pred, label_target)
        bb_loss = F.mse_loss(bb_preds, bb_targets)
        loss = (clf_loss + 0.001*bb_loss)

        return loss, clf_loss, bb_loss
