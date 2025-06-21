import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossMulticlass(nn.Module):
    def __init__(self, weight=None, gamma=2.0, ignore_index=255, reduction='mean'):
        # weight: tensor for class weights to balance classes
        # gamma: focusing parameter to reduce loss for easy examples (default 2.0)
        # ignore_index: label to ignore in loss computation 
        # reduction: method to reduce loss -> mean
        super(FocalLossMulticlass, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        # Reshape input and target for pixel-wise loss calculation
        if input.dim() > 2:
            input = input.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
            input = input.view(-1, input.size(-1))          # [B*H*W, C]
            target = target.view(-1)                        # [B*H*W]

        #Compute log probabilities for each class base on PREDICTION:
        #log_probs[i][j] is the logarithm of the probability that pixel i belongs to class j, 
        #and it's computed using softmax
        log_probs = F.log_softmax(input, dim=1)
        # Convert to normal probabilities
        probs = log_probs.exp()
        
        # Mask to ignore certain labels (ignore_index)
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        input = input[valid_mask]
        log_probs = log_probs[valid_mask]
        probs = probs[valid_mask]
    
        #Compute log probabilities for each class base on GROUND TRUTH:
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()


        if self.weight is not None:
            if self.weight.device != input.device:
                self.weight = self.weight.to(input.device)
            #Multiply the log-probability by the weight of the correct class. So rare classes contribute more to the loss.
            at = self.weight.gather(0, target)
            log_pt = log_pt * at

        # Calculate Focal loss -> log_pt is already given by log_pt * at
        loss = -((1 - pt) ** self.gamma) * log_pt


        #Reduce the loss -> In line with standard practices to improve comparability and stability during training,
        #the final loss was averaged over the number of examples.
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
