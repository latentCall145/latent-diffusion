import sys
sys.path.append('.') # for python ./scripts/file.py
sys.path.append('..') # for python file.py within ./scripts dir

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF']='backend:cudaMallocAsync'
#os.environ['CUDA_LAUNCH_BLOCKING']='1'
from modules.models import VAE, UNet, UNetSD, UNetSDXL
import torch, time

with torch.no_grad():
    with torch.amp.autocast('cuda', dtype=torch.float16):
        #model = UNet()
        #model = UNetSD()
        #model = UNetSDXL()
        model = VAE(ch_mults=[1,2,4,4])
        #print(model)
        print(model)
        params = sum([p.numel() for p in model.parameters()])
        print(f'Params (in millions): {params/1e6}')
        #model = model.half().to(0)
        raise
        model = model.to(0)

        is_unet = isinstance(model, UNet)
        spatial_mult = 1 if is_unet else 8
        in_c = model.in_c if is_unet else 3
        context_dim = model.context_dim if is_unet else 1
        x = torch.zeros((1, in_c, 32*spatial_mult, 32*spatial_mult)).to(0) #.half()

        t = torch.zeros((1,)).long().to(0)
        c = torch.zeros((1, 64, context_dim)).to(0) #.half()

        tic = time.time()
        model = torch.compile(model)

        for _i in  range(2):
            _ = model(x, t, c) if is_unet else model(x)
            torch.cuda.synchronize()
        print('Time to compile model:', time.time() - tic)
        print('alloc:', torch.cuda.max_memory_allocated() / (1024**3))
        print('reser:', torch.cuda.max_memory_reserved() / (1024**3))

        tic = time.time()
        N = 20
        for i in range(N):
            print(i, end='\r')
            _ = model(x, t, c) if is_unet else model(x)
            torch.cuda.synchronize()
        print('Time per step (ms):', (time.time() - tic) * 1000  / N)
