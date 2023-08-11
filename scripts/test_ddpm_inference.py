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
with torch.no_grad():
    with torch.amp.autocast(device):
        homedir = os.path.expanduser('~')
        img_fname = f'{homedir}/Projects/test.jpg'
        x1 = load_img(img_fname).to(device)

        sd = torch.load('../trained_models/vae_0003.pth')['vae_state_dict']
        vae = VAE()
        vae.load_state_dict(sd, strict=True)
        vae = vae.to(device)
        vae.eval()

        ddpm = UNet()
        sd = torch.load('../trained_models/ddpm_0037.pth')['ddpm_state_dict']
        ddpm.load_state_dict(sd, strict=True)
        ddpm = ddpm.to(device)
        ddpm.eval()
        #ddpm = torch.compile(ddpm)

        nsr = NoiseSchedule().to(device)
        sampler = DDIMSampler(ddpm, nsr, tau_dim=20)

        #mean, _log_var = torch.split(vae.encoder(x1), vae.nz, dim=1)
        #mean /= mean.std()
        #for img_fname in ['test.jpg', 'deer256.jpg', 'cake.jpg']:
        #    mean, _log_var = torch.split(vae.encoder(load_img(f'{homedir}/Projects/{img_fname}').to(device)), vae.nz, dim=1)
        #    mean /= mean.std()
        #    noise = torch.randn_like(mean)
        #    ts, losses = [], []
        #    for t in range(1, 1000, 100):
        #        noised = mean * nsr.signal_stds[t] + noise * nsr.noise_stds[t]
        #        noise_pred = ddpm(noised, torch.tensor([t], dtype=torch.long, device=device))
        #        loss = F.mse_loss(noise_pred, noise).item()
        #        print('T:', t, 'MSE:', loss)
        #        ts.append(t)
        #        losses.append(loss)
        #    plt.plot(ts, losses)
        #plt.show()

        #fig, ax = plt.subplots(2, 2)
        #ax[0][0].imshow(tn(mean)[:, :, :3])
        #ax[0][1].imshow(tn(noise)[:, :, :3])
        #ax[1][0].imshow(tn(noised)[:, :, :3])
        #ax[1][1].imshow(tn(noise_pred)[:, :, :3])
        #plt.show()
        #raise

        text_embedder = FrozenCLIPEmbedder()

        while True:
            prompt = input('Enter prompt: ')
            initial_x = torch.randn((1, 4, 32, 32), device=device)

            tic = time.time()
            text_embedding = text_embedder([prompt])
            print(f'Time for CLIP inference: {time.time()-tic:.4f} s')

            tic = time.time()
            z = sampler.get_samples(initial_x=initial_x, context=text_embedding, cfg_weight=5)
            print(f'Time for DM inference of {initial_x.shape} shape tensor: {time.time()-tic:.4f} s')

            tic = time.time()
            ypred = vae.decoder(z)
            torch.cuda.synchronize()
            print(f'Time for VAE decoder inference of {z.shape} shape tensor: {time.time()-tic:.4f} s')
            plt.imshow(tn(ypred))
            plt.show()

            #fig, ax = plt.subplots(1, 2)
            #ax[0].set_title('Input')
            #ax[0].imshow(tn(x1))
            #ax[1].set_title('VAE Reconstruction')
            #ax[1].imshow(tn(ypred))
            #plt.show()
