# taken from https://huggingface.co/blog/stable_diffusion
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import numpy as np
import torch

H, W = 512, 512
CFG_SCALE = 7.5
N_STEPS = 50
#prompts = ["a single scull rowing into the sunset, beautiful, dslr photograph 4k, high contrast"] * 1
prompts = ["white image"] * 1

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
pipe.unet = pipe.unet.half()
pipe.to("cuda")

images = pipe(prompts, guidance_scale=CFG_SCALE, num_inference_steps=N_STEPS, height=H, width=W).images

for p, image in zip(prompts, images):
    plt.imshow(image)
    plt.title(p)
    plt.show()
    plt.close('all')
