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
        model = UNet()
        #model = UNetSD()
        #model = UNetSDXL()
        #model = VAE().to(0)
        print(model)
        params = sum([p.numel() for p in model.parameters()])
        print(f'Params (in millions): {params/1e6}')
        raise
        model = model.half().to(0)
        model = torch.compile(model)
        x = torch.zeros((1, model.in_c, 64, 64)).to(0) #.half()
        c = torch.zeros((1, 77, model.context_dim)).to(0) #.half()
        t = torch.zeros((1,)).long().to(0)

        for _ in  range(2):
            model(x, t, c)
            torch.cuda.synchronize()
        print('alloc:', torch.cuda.max_memory_allocated() / (1024**3))
        print('reser:', torch.cuda.max_memory_reserved() / (1024**3))

        tic = time.time()
        N = 20
        for i in range(N):
            print(i, end='\r')
            model(x, t, c)
            torch.cuda.synchronize()
        print((time.time() - tic) * 1000  / N)
