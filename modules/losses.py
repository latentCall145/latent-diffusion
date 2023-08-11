from modules.models import Discriminator
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

def kl_divergence(mean, logvar):
    return 0.5 * torch.sum(mean.pow(2) + logvar.exp() - logvar - 1) / mean.shape[0]

def adv_loss_fn(logits_real, logits_fake):
    loss_real = F.relu(1. - logits_real)
    loss_fake = F.relu(1. + logits_fake)
    d_loss = 0.5 * (loss_real + loss_fake).mean()
    return d_loss
