import sys
sys.path.append('.') # for python ./scripts/file.py
sys.path.append('..') # for python file.py within ./scripts dir

from ptflops import get_model_complexity_info
from torch import nn

from modules.models import FrozenCLIPEmbedder, VAE, UNet
from transformers import CLIPTokenizerFast, CLIPTextModel, logging
logging.set_verbosity_error()

from torchvision.models import vgg16, VGG16_Weights
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

def get_gmacs(macs_str):
    macs, base = macs_str.split()
    macs = float(macs)
    if 'kmac' in base.lower():
        return macs / 1000 / 1000
    elif 'mmac' in base.lower():
        return macs / 1000
    elif 'gmac' in base.lower():
        return macs

# v1 (vae - nc 64, clip - 77 tokens, unet - 16x16x16 latent, no swiglu): 383.22 gflops forward + backward pass
# v2 (vae - nc 128, clip - 64 tokens, unet - 32x32x4 latent, swiglu): 1414.5 gflops forward + backward pass

net = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
clip_input_constructor = lambda shape: {'input_ids': torch.ones((1, *shape), dtype=torch.long)}
clip_macs, params = get_model_complexity_info(net, (64,), input_constructor=clip_input_constructor, as_strings=True, print_per_layer_stat=False, verbose=False)
print('CLIP')
print('{:<30}  {:<8}'.format('Computational complexity: ', clip_macs))
print('{:<30}  {:<8}'.format('N params: ', params))

net = VAE()
#net = VAE(64, [1, 2, 2, 4], nz=16)
vae_macs, params = get_model_complexity_info(net, (3,256,256), as_strings=True, print_per_layer_stat=False, verbose=False)
print('VAE')
print('{:<30}  {:<8}'.format('Computational complexity: ', vae_macs))
print('{:<30}  {:<8}'.format('N params: ', params))

net = UNet() #attn_resolutions=[0,2,10])
#net = UNet(in_c=16, nc=256, ch_mults=[1,2,4], attn_resolutions=[1,1,1])
unet_input_constructor = lambda shape: {'x': torch.randn(1, *shape), 'timesteps': torch.zeros((1,), dtype=torch.long), 'context': torch.zeros((1, 64, 768))}
unet_macs, params = get_model_complexity_info(net, (net.in_c, 32, 32), input_constructor=unet_input_constructor, as_strings=True, print_per_layer_stat=False, verbose=False)
print('UNet')
print('{:<30}  {:<8}'.format('Computational complexity: ', unet_macs))
print('{:<30}  {:<8}'.format('N params: ', params))

total_gmacs = sum(get_gmacs(i) for i in (clip_macs, vae_macs, unet_macs))
print('Total GMacs:', total_gmacs)
print('Total GFLOPs per forward pass:', 2*total_gmacs)
print('Total GFLOPs per forward and backward pass:', 3*2*total_gmacs)
