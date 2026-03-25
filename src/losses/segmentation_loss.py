import torch
import torch.nn as nn

def focal_loss(pred, mask, gamma=2.0):
    p = mask*pred + (1-mask)*(1-pred)
    loss = ((1-p)**gamma * (-torch.log(p+1e-8))).mean()
    return loss

def l1_loss(pred, mask):
    return torch.abs(pred-mask).mean()

def segmentation_loss(pred, mask, gamma=2.0):
    return focal_loss(pred, mask, gamma) + l1_loss(pred, mask)
