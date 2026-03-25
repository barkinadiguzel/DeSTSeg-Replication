import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def generate_anomaly(I_n, A, M, beta_range=(0.15,1.0)):
    beta = random.uniform(*beta_range)
    Ia = beta * (M*A) + (1-beta)*(M*I_n) + (1-M)*I_n
    return Ia
