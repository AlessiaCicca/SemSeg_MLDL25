import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossMulticlass(nn.Module):
    def __init__(self, weight=None, gamma=2.0, ignore_index=255, reduction='mean'):
        """
        Focal Loss per classificazione multi-classe.

        Args:
            weight (Tensor): pesi per classe [C], opzionali.
            gamma (float): parametro gamma (>0).
            ignore_index (int): etichetta da ignorare nel calcolo della loss.
            reduction (str): 'mean', 'sum', 'none'.
        """
        super(FocalLossMulticlass, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        Args:
            input: logits grezzi, shape [B, C, H, W]
            target: etichette intere, shape [B, H, W]
        """
        if input.dim() > 2:
            input = input.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
            input = input.view(-1, input.size(-1))          # [B*H*W, C]
            target = target.view(-1)                        # [B*H*W]

        log_probs = F.log_softmax(input, dim=1)
        probs = log_probs.exp()

        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        input = input[valid_mask]
        log_probs = log_probs[valid_mask]
        probs = probs[valid_mask]

        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        # Pesi per classe
        if self.weight is not None:
            if self.weight.device != input.device:
                self.weight = self.weight.to(input.device)
            at = self.weight.gather(0, target)
            log_pt = log_pt * at

        loss = -((1 - pt) ** self.gamma) * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
