import torch
import torch.nn.functional as F

def cosine_similarity(T_feat, S_feat):
    B,C,H,W = T_feat.shape
    T_flat = T_feat.permute(0,2,3,1).reshape(-1,C)  
    S_flat = S_feat.permute(0,2,3,1).reshape(-1,C)
    cos = F.cosine_similarity(T_flat, S_flat, dim=1)  
    D = 1 - cos
    return D.view(B,H,W)
