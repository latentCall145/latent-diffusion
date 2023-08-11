import sys
sys.path.append('.') # for python ./scripts/file.py
sys.path.append('..') # for python file.py within ./scripts dir

from modules.samplers import NoiseSchedule, DDIMSampler
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch, math, os

def denorm(x):
    mean = torch.tensor([0.485, 0.456, 0.406])[None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, None]
    return x * std + mean

def norm(x):
    mean = torch.tensor([0.485, 0.456, 0.406])[None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, None]
    return (x - mean) / std

img = np.array(Image.open(os.path.expanduser('~') + '/Projects/test.jpg')).astype(np.float32) / 255.
x = norm(torch.from_numpy(img))
z = torch.randn_like(x)

nsr = NoiseSchedule()
noised = z

print('Initial')
steps = 10
samp = DDIMSampler(None)
print(samp.tau)

print(x.mean(), x.std())
print('noised_T:', noised.mean(), noised.std())
fig, ax = plt.subplots(1,2)
ax[0].imshow(denorm(x))
ax[1].imshow(denorm(noised))
plt.show()
noise_pred = (noised - nsr.signal_stds[-1]*x) / nsr.noise_stds[-1]
for t in range(steps, 0, -1):
    noise_pred = (noised - nsr.signal_stds[samp.tau[t-1]]*x) / nsr.noise_stds[samp.tau[t-1]]

    print(f'T: {t}, Tau[t-1]: {samp.tau[t-1]}')
    noised = samp.denoise_step(noised, t, noise_pred)
    if t % (steps//10) == 0:
        print(f'noised_{t-1}', noised.mean(), noised.std())
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(denorm(x))
        ax[1].imshow(denorm(noised))
        plt.show()

print('Final')
print(f'end noised_{t-1}', noised.mean(), noised.std())
fig, ax = plt.subplots(1,2)
ax[0].imshow(denorm(x))
ax[1].imshow(denorm(noised))
plt.show()
