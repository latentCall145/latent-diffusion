import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch, math, os
try:
    from flash_attn import flash_attn_func
    ENABLE_FLASH_ATTN = ('DISABLE_FLASH' not in os.environ) or (os.environ['DISABLE_FLASH'] != '1')
except:
    ENABLE_FLASH_ATTN = False

if ENABLE_FLASH_ATTN:
    print('Flash Attention enabled (note: may be slower as Flash Attention only gets speedups with large inputs).')
else:
    print('Flash Attention not enabled.')

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class ResBlock(nn.Module):
    def __init__(self, in_c: int, nc: int, temb_c: int = None):
        '''
        in_c: number of input channels
        nc: number of output channels
        temb_c: number of t (time?) embedding input channels (or None if no time embedding)
        '''
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_c, eps=1e-4)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_c, nc, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, nc, eps=1e-4)
        self.act2 = nn.SiLU()
        self.conv2 = zero_module(nn.Conv2d(nc, nc, 3, padding=1))
        if temb_c is not None:
            self.temb_proj = nn.Linear(temb_c, nc)
        self.skip = nn.Conv2d(in_c, nc, 1, bias=False) if in_c != nc else None
    
    def forward(self, x, temb=None): # temb = t (time) embedding
        skip = x if self.skip is None else self.skip(x)
        x = self.conv1(self.act1(self.norm1(x)))
        if temb is not None:
            x = x + self.temb_proj(F.silu(temb))[:, :, None, None]
        x = self.conv2(self.act2(self.norm2(x)))

        return x + skip

class Downsample(nn.Module):
    def __init__(self, nc: int):
        '''
        nc: number of input and output channels
        '''
        super().__init__()
        self.layer = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(nc, nc, 3, 2))

    def forward(self, x):
        return self.layer(x)

class Upsample(nn.Module):
    def __init__(self, nc: int):
        '''
        nc: number of input and output channels
        '''
        super().__init__()
        self.conv = nn.Conv2d(nc, nc, 3, padding=1)

    def forward(self, x):
        _B, _C, H, W = x.shape
        return self.conv(F.interpolate(x, size=(H*2, W*2), mode='nearest')) # specifying scale factor offloads op to CPU: https://github.com/pytorch/xla/issues/2588

class MHA(nn.Module): # slightly faster and less mem than torch multihead attn (I suppose from QKV projection being fused)
    def __init__(self, nc: int, nh: int, kv_dim: int = None, zero_last_layer: bool = True):
        '''
        nc: number of input and output channels
        nh: number of heads (note: d_head = nc // nh)
        kv_dim: dimensionality of key & value input (used for conditioning input in cross-attention; self-attention if kv_dim is None)
        zero_last_layer: whether or not to zero-init the weights of the last layer (this helps out optimization of residual connections)
        '''
        super().__init__()
        self.nh = nh
        self.dhead = nc // nh
        
        kv_dim = nc if kv_dim is None else kv_dim
        self.q_in = nn.Linear(nc, nc, bias=False)
        self.k_in = nn.Linear(kv_dim, nc, bias=False)
        self.v_in = nn.Linear(kv_dim, nc, bias=False)
        self.out = nn.Linear(nc, nc, bias=False)
        if zero_last_layer:
            self.out = zero_module(self.out)
    
    def split_heads(self, x):
        B, L, E = x.shape
        if ENABLE_FLASH_ATTN:
            return x.reshape(B, L, self.nh, self.dhead) # M N (H D) -> M N H D, D=self.dhead, H=self.nh)
        return x.reshape(B, L, self.nh, self.dhead).permute(0, 2, 1, 3).contiguous() # M N (H D) -> M H N D, D=self.dhead, H=self.nh
    
    def forward(self, q, kv=None):
        B, L, E = q.shape
        if kv is None:
            q, k, v = map(self.split_heads, (self.q_in(q), self.k_in(q), self.v_in(q)))
        else:
            q, k, v = map(self.split_heads, (self.q_in(q), self.k_in(kv), self.v_in(kv)))

        if ENABLE_FLASH_ATTN:
            qkv = flash_attn_func(q, k, v) # flash attention not on TPU
            concatted = qkv.reshape(B, L, E) # M N H D -> M N (H D)
        else:
            qkv = F.scaled_dot_product_attention(q, k, v) # flash attention not on TPU
            concatted = qkv.permute(0, 2, 1, 3).reshape(B, L, E).contiguous() # M H N D -> M N (H D)
        return self.out(concatted)
    
class Attn2d(nn.Module):
    def __init__(self, nc: int):
        '''
        nc: number of input and output channels
        '''
        super().__init__()
        self.nc = nc
        self.norm = nn.GroupNorm(32, self.nc, eps=1e-4)
        self.attn = MHA(self.nc, max(self.nc // 128, 1)) # max head dim is 128

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H*W).permute(0, 2, 1) # B C H W -> B (H W) C
        h = self.attn(h)
        h = h.permute(0, 2, 1).reshape(B, C, H, W) # B (H W) C -> B C H W
        return x + h

class SwiGLU(nn.Module):
    def __init__(self, in_c: int, nc: int, bias: bool = False):
        super().__init__()
        self.lin = nn.Linear(in_c, nc, bias=bias)
        self.gate = nn.Linear(in_c, nc, bias=bias)

    def forward(self, x):
        return self.lin(x) * F.silu(self.gate(x))

class TransformerBlock(nn.Module):
    '''
    Attention between UNet feature maps and text embeddings.
    '''
    def __init__(self, d: int, d_emb: int, nh: int = 8):
        '''
        d: attention dimensionality
        d_emb: dimensionality of conditioning embedding (text conditioning)
        nh: number of heads
        '''
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)
        self.attn1 = MHA(d, nh)
        self.attn2 = MHA(d, nh, kv_dim=d_emb)
        self.ff = nn.Sequential(
            SwiGLU(d, 4*d, bias=False),
            nn.Linear(4*d, d, bias=False),
        )

    def forward(self, x, context=None):
        B, C, H, W = x.shape
        skip = x
        x = x.reshape(B, C, H*W).permute(0, 2, 1).contiguous() # B C H W -> B (H W) C
        x = self.attn1(self.norm1(x)) + x
        if context is not None:
            x = self.attn2(self.norm2(x), context) + x
        x = self.ff(self.norm3(x)) + x
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous() # B (H W) C -> B C H W

        return x + skip

class TimeEmbedding(nn.Module):
    '''
    Sinusoidal time embedding with a feed forward network.
    '''
    def __init__(self, embed_c: int, out_c: int, max_period: int = 10000):
        '''
        embed_c: dimensionality of sinusoidal time embedding
        out_c: dimensionality of projected (output) embedding
        max_period: controls the minimum frequency of the embeddings
        '''
        super().__init__()
        self.embed_c = embed_c
        self.out_c = out_c
        self.max_period = max_period
        half = embed_c // 2
        self.register_buffer('freqs', torch.exp(-math.log(max_period) * torch.linspace(0, 1, half)))
        self.ff = nn.Sequential(
            nn.Linear(embed_c, out_c),
            nn.SiLU(),
            nn.Linear(out_c, out_c),
        )

    def forward(self, timesteps):
        t_freqs = timesteps[:, None] * self.freqs[None, :]
        emb = torch.cat([t_freqs.cos(), t_freqs.sin()], dim=-1)
        emb = self.ff(emb)
        return emb
