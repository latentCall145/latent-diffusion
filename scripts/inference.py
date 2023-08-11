import sys
sys.path.append('.') # for python ./scripts/file.py
sys.path.append('..') # for python file.py within ./scripts dir
from modules.models import VAE, FrozenCLIPEmbedder, UNet
from modules.samplers import NoiseSchedule, DDIMSampler
from transformers import logging
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch, glob, os, gc

torch.backends.cudnn.benchmark = True
logging.set_verbosity_error() # disables the "Some weights failed to load" for the CLIP text model (non-issue, these failed weights are for the ViT and aren't used anyway)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

USE_MIXED = True
DDPM_CKPT_PATH = 'ddpm_0080.pth'
VAE_CKPT_PATH = 'vae_0090.pth'

ddpm = UNet().to('cuda')
vae = VAE().eval().to('cuda')
text_embedder = FrozenCLIPEmbedder(version="openai/clip-vit-large-patch14", device='cuda')
noise_schedule = NoiseSchedule().to('cuda')

print(f'Loading VAE checkpoint from {VAE_CKPT_PATH}')
vae_ckpt = torch.load(VAE_CKPT_PATH)
vae.load_state_dict(vae_ckpt['vae_state_dict'], strict=True)

'''
with torch.no_grad():
    x = torch.from_numpy(np.array(Image.open('../test.jpg'))).float().permute(2, 0, 1)[None] / 127.5 - 1.0
    y, _, _ = vae(x.cuda())
    plt.imshow(y[0].permute(1, 2, 0).cpu()*0.5+0.5); plt.show()
'''

print(f'Loading checkpoint from {DDPM_CKPT_PATH}')
ckpt = torch.load(DDPM_CKPT_PATH)
ddpm.load_state_dict(ckpt['ddpm_state_dict'], strict=True)
sampler = DDIMSampler(ddpm, noise_schedule=noise_schedule, tau_dim=50)

N_ROWS, N_COLS = 2, 5
n_imgs = N_ROWS * N_COLS
captions = ['bright day, dslr photograph'] * 10
assert len(captions) == n_imgs

text_embeddings = text_embedder(captions)

z = torch.randn((n_imgs, 16, 16, 16), device='cuda')

with torch.no_grad():
    with torch.autocast('cuda', enabled=USE_MIXED):
        z_denoised = sampler.get_samples(initial_x=z, context=text_embeddings)
        preds = vae.decoder(z_denoised)

preds = preds.permute(0, 2, 3, 1).detach().float().cpu().clamp(-1, 1) * 0.5 + 0.5
fig, ax = plt.subplots(N_ROWS, N_COLS, figsize=(20, 10))

for i in range(n_imgs):
    row, col = i // N_COLS, i % N_COLS
    if N_ROWS == N_COLS == 1:
        ax.title.set_text(captions[i])
        ax.imshow(preds[i])
    elif N_ROWS == 1 or N_COLS == 1:
        ax[i].title.set_text(captions[i])
        ax[i].imshow(preds[i])
    else:
        ax[row][col].title.set_text(captions[i])
        ax[row][col].imshow(preds[i])
plt.show()
