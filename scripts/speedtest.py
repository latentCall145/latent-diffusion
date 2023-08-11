import sys
sys.path.append('.') # for python ./scripts/file.py
sys.path.append('..') # for python file.py within ./scripts dir
from modules.models import VAE, VAES
import numpy as np
import torch, os, time

#torch.set_default_tensor_type(torch.HalfTensor)
device = 'cuda'
x = torch.randn((16, 3, 256, 256), dtype=torch.half).to(device)
with torch.no_grad():
    for M in (VAE, VAES):
        print('Model type:', M)
        model = M()
        model = torch.compile(model)
        model = model.to(device).half()
        for _ in range(2):
            model(x)
            torch.cuda.synchronize()
        print('alloc:', torch.cuda.max_memory_allocated() / (1024**3))
        print('reser:', torch.cuda.max_memory_reserved() / (1024**3))

        tic = time.time()
        N = 10
        for i in range(N):
            print(i, end='\r')
            model(x)
            torch.cuda.synchronize()
        print((time.time() - tic) * 1000  / N)
