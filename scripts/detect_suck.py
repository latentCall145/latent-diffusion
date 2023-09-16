import sys
sys.path.append('.') # for python ./scripts/file.py
sys.path.append('..') # for python file.py within ./scripts dir

from modules.models import VAE, UNet, FrozenCLIPEmbedder
from modules.samplers import NoiseSchedule, DDIMSampler
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
    #print(f'mean: {x.mean()}, std: {x.std()}')
    return x

def tn(tensor, idx=0):
    if idx is None:
        return tensor.cpu().permute(0, 2, 3, 1).numpy()
    return tensor.float().cpu().clamp(-1, 1).permute(0, 2, 3, 1).numpy()[idx] * 0.5 + 0.5

device = 'cuda'
nsr = NoiseSchedule().to(device)

def get_loss_curve(img, ddpm, timesteps): # track DM's MSE loss of an image through time
    mean, log_var = torch.split(vae.encoder(img.to(device)), vae.nz, dim=1)
    z = vae.sample(mean, log_var)
    z /= z.std()
    print('Z:', z.mean(), z.std())
    noise = torch.randn_like(z)
    losses = []
    for t_idx, t in enumerate(timesteps):
        noised = z * nsr.signal_stds[t] + noise * nsr.noise_stds[t]
        print('X_t:', noised.mean(), noised.std())
        noise_pred = ddpm(noised, torch.tensor([t], dtype=torch.long, device=device))
        print('noise prediction:', noise_pred.abs().mean(), noise_pred.std())

        loss = F.mse_loss(noise_pred, noise).item()
        losses.append(loss)
    return losses

with torch.no_grad():
    with torch.amp.autocast(device, enabled=False):
        homedir = os.path.expanduser('~')
        img_fname = f'{homedir}/Projects/test.jpg'
        x1 = load_img(img_fname).to(device)

        sd = torch.load('../trained_models/vae_0021.pth')['vae_state_dict']
        vae = VAE()
        vae.load_state_dict(sd)
        vae = vae.to(device)
        vae.eval()

        ddpm = UNet()
        sd = torch.load(f'../trained_models/ddpm_0008_dead.pth')['ddpm_state_dict']
        ddpm.load_state_dict(sd)
        ddpm = ddpm.to(device)
        ddpm.eval()
        
        nsr = nsr.to(device)
        sampler = DDIMSampler(ddpm, nsr, tau_dim=20)

        img_fnames = ['test.jpg', 'deer256.jpg', 'cake.jpg']
        colors = ['blue', 'red', 'green', 'orange', 'pink', 'black'][:len(img_fnames)]

        for img_fname, color in zip(img_fnames, colors):
            losses = get_loss_curve(load_img(f'{homedir}/Projects/{img_fname}'), ddpm, sampler.tau)
            ts = range(len(losses))
            print(ts)
            print(losses)
            plt.plot(ts, losses, label=f'{img_fname}', color=color)
        plt.legend()
        plt.show()

        raise

        text_embedder = FrozenCLIPEmbedder()

        while True:
            prompt = input('Enter prompt: ')
            initial_x = torch.randn((1, 4, 32, 32), device=device)

            tic = time.time()
            text_embedding = text_embedder([prompt])
            print(f'Time for CLIP inference: {time.time()-tic:.4f} s')

            tic = time.time()
            z = sampler.get_samples(initial_x=initial_x, context=text_embedding, cfg_weight=5) * 1.2
            print(f'Time for DM inference of {initial_x.shape} shape tensor: {time.time()-tic:.4f} s')

            tic = time.time()
            ypred = vae.decoder(z)
            torch.cuda.synchronize()
            print(f'Time for VAE decoder inference of {z.shape} shape tensor: {time.time()-tic:.4f} s')
            Image.fromarray((255*tn(ypred)).astype(np.uint8)).save('/tmp/ldm_img.png')
            plt.imshow(tn(ypred))
            plt.show()
