import torch

def cosine_loss(T_feats, S_feats):
    from ..layers.similarity import cosine_similarity
    loss = 0
    for T,S in zip(T_feats, S_feats):
        D = cosine_similarity(T,S)
        loss += D.mean()
    return loss
