def compute_multilevel_similarity(T_feats, S_feats):
    from ..layers.similarity import cosine_similarity
    X_list = []
    for T,S in zip(T_feats, S_feats):
        X_list.append(cosine_similarity(T,S).unsqueeze(1))  
    return torch.cat(X_list, dim=1)  
