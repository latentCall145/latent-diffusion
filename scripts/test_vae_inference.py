import sys
sys.path.append('.') # for python ./scripts/file.py
sys.path.append('..') # for python file.py within ./scripts dir

from modules.models import VAE, Discriminator
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch, time, os

def load_img(img_fname, size=None):
    img = Image.open(img_fname)
    if size is not None:
        img = img.resize(size)
    x = np.array(img)
    x = (torch.from_numpy(x).permute(2, 0, 1)[None] / 127.5 - 1.0)
    print(f'mean: {x.mean()}, std: {x.std()}')
    return x

def tn(tensor, idx=0):
    if idx is None:
        return tensor.cpu().permute(0, 2, 3, 1).numpy()
    return tensor.float().cpu().clamp(-1, 1).permute(0, 2, 3, 1).numpy()[idx] * 0.5 + 0.5

def show_random_sample(mean, log_var):
    std = torch.exp(0.5 * log_var).to(device)
    eps = torch.randn_like(log_var).to(device)
    z = mean + eps * std
    out = vae.decoder(z)
    plt.imshow(tn(out))
    plt.show()

with torch.no_grad():
    device = 'cuda'
    homedir = os.path.expanduser('~')
    img_fname = f'{homedir}/Projects/cake.jpg'
    x1 = load_img(img_fname).to(device).half()

    sd = torch.load('../trained_models/vae_0003.pth')['vae_state_dict']
    vae = VAE().half()
    vae.load_state_dict(sd, strict=True)
    vae = vae.to(device)
    vae.eval()

    tic = time.time()
    ypred, _mean, _log_var = vae(x1)
    torch.cuda.synchronize()
    print(f'Time for VAE inference of {x1.shape} shape tensor: {time.time()-tic:.4f} s')

    fig, ax = plt.subplots(2,2)
    for i in range(4):
        plotted = ax[i//2][i%2].imshow(_mean.cpu().numpy()[0, i])
        plt.colorbar(plotted, ax=ax[i//2][i%2])
    plt.show()

    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('Input')
    ax[0].imshow(tn(x1))
    ax[1].set_title('VAE Reconstruction')
    ax[1].imshow(tn(ypred))
    plt.show()
