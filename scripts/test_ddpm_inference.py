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
        sd = torch.load(f'../trained_models/ddpm_0102_fixed.pth')['ddpm_state_dict']
        ddpm.load_state_dict(sd, strict=True)
        ddpm = ddpm.to(device)
        ddpm.eval()
        
        nsr = NoiseSchedule().to(device)
        sampler = DDIMSampler(ddpm, nsr, tau_dim=20)

        #samp_models = [37, 64, 102]
        #opacities = np.linspace(0, 1, len(samp_models)+1)[1:]
        #img_fnames = ['test.jpg', 'deer256.jpg', 'cake.jpg']
        #colors = ['blue', 'red', 'green', 'orange', 'pink', 'black'][:len(img_fnames)]
        #for mv, opacity in zip(samp_models, opacities):
        #    sd = torch.load(f'../trained_models/ddpm_{mv:0>4}.pth')['ddpm_state_dict']
        #    ddpm.load_state_dict(sd, strict=True)
        #    ddpm = ddpm.to(device)
        #    ddpm.eval()
        #    #ddpm = torch.compile(ddpm)

        #    nsr = NoiseSchedule().to(device)
        #    sampler = DDIMSampler(ddpm, nsr, tau_dim=50)

        #    #mean, _log_var = torch.split(vae.encoder(x1), vae.nz, dim=1)
        #    #mean /= mean.std()
        #    for img_fname, color in zip(img_fnames, colors):
        #    #for img_fname in ['cake.jpg']:
        #        mean, log_var = torch.split(vae.encoder(load_img(f'{homedir}/Projects/{img_fname}').to(device)), vae.nz, dim=1)
        #        for _ in range(1):
        #            z = vae.sample(mean, log_var)
        #            z /= z.std()
        #            noise = torch.randn_like(z)
        #            ts, losses = [], []
        #            #fig, ax  = plt.subplots(4, 5)
        #            for t_idx, t in enumerate(sampler.tau):
        #                noised = z * nsr.signal_stds[t] + noise * nsr.noise_stds[t]
        #                noise_pred = ddpm(noised, torch.tensor([t], dtype=torch.long, device=device))
        #                #ax[t_idx//5][t_idx%5].hist(noise_pred.flatten().cpu().numpy())
        #                #ax[t_idx//5][t_idx%5].set_title(f'T: {t}, std: {noise_pred.std()}')
        #                #ax[2+t_idx//5][t_idx%5].hist(noise_pred.flatten().cpu().numpy())
        #                #ax[2+t_idx//5][t_idx%5].set_title(f'z; T: {t}, std: {z.std()}')
        #                loss = F.mse_loss(noise_pred, noise).item()
        #                print('T:', t, 'MSE:', loss, noise_pred.std(), noise.std())
        #                ts.append(t)
        #                losses.append(loss)
        #            plt.plot(ts, losses, label=f'{mv}_{img_fname}', color=color, alpha=opacity)
        #            #plt.show()
        #plt.legend()
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
            Image.fromarray((255*tn(ypred)).astype(np.uint8)).save('/tmp/ldm_img.png')
            plt.imshow(tn(ypred))
            plt.show()
