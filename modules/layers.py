import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch, math, os, einops
try:
    from flash_attn import flash_attn_func
    ENABLE_FLASH_ATTN = ('DISABLE_FLASH' not in os.environ) or (os.environ['DISABLE_FLASH'] != '1')
except:
    ENABLE_FLASH_ATTN = False

if ENABLE_FLASH_ATTN:
    print('Flash Attention enabled (note: may be slower as Flash Attention only gets speedups with large inputs).')
else:
    print('Flash Attention not enabled.')

rearrange = lambda tensor, pattern, **axes_lengths: einops.rearrange(tensor, pattern, **axes_lengths).contiguous()

class ResBlock(nn.Module):
    def __init__(self, in_c: int, nc: int, temb_c: int | None = None):
        '''
        in_c: number of input channels
        nc: number of output channels
        temb_c: number of t (time?) embedding input channels (or None if no time embedding)
        '''
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_c)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_c, nc, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, nc)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(nc, nc, 3, padding=1)
        if temb_c is not None:
            self.temb_proj = nn.Linear(temb_c, nc)
        self.skip = nn.Conv2d(in_c, nc, 1) if in_c != nc else None
    
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
        return self.conv(F.interpolate(x, scale_factor=2.0, mode='nearest'))

class MHA(nn.Module): # slightly faster and less mem than torch multihead attn (I suppose from QKV projection being fused)
    def __init__(self, nc: int, nh: int, kv_dim: int | None = None):
        '''
        nc: number of input and output channels
        nh: number of heads (note: d_head = nc // nh)
        kv_dim: dimensionality of key & value input (used for conditioning input in cross-attention; self-attention if kv_dim is None)
        '''
        super().__init__()
        self.nh = nh
        self.dhead = nc // nh
        
        kv_dim = nc if kv_dim is None else kv_dim
        self.q_in = nn.Linear(nc, nc, bias=False)
        self.k_in = nn.Linear(kv_dim, nc, bias=False)
        self.v_in = nn.Linear(kv_dim, nc, bias=False)
        self.out = nn.Linear(nc, nc, bias=False)
    
    def split_heads(self, x):
        B, L, E = x.shape
        if ENABLE_FLASH_ATTN:
            #return rearrange(x, 'M N (H D) -> M N H D', D=self.dhead, H=self.nh)
            return x.reshape(B, L, self.nh, self.dhead)
        #return rearrange(x, 'M N (H D) -> M H N D', D=self.dhead, H=self.nh)
        return x.reshape(B, L, self.nh, self.dhead).permute(0, 2, 1, 3).contiguous()
    
    def forward(self, q, kv=None):
        B, L, E = q.shape
        if kv is None:
            q, k, v = map(self.split_heads, (self.q_in(q), self.k_in(q), self.v_in(q)))
        else:
            q, k, v = map(self.split_heads, (self.q_in(q), self.k_in(kv), self.v_in(kv)))

        if ENABLE_FLASH_ATTN:
            qkv = flash_attn_func(q, k, v) # flash attention not on TPU
            #concatted = rearrange(qkv, 'M N H D -> M N (H D)')
            concatted = qkv.reshape(B, L, E)
        else:
            qkv = F.scaled_dot_product_attention(q, k, v) # flash attention not on TPU
            #concatted = rearrange(qkv, 'M H N D -> M N (H D)')
            concatted = qkv.permute(0, 2, 1, 3).reshape(B, L, E).contiguous()
        return self.out(concatted)
    
class Attn2d(nn.Module):
    def __init__(self, nc: int):
        '''
        nc: number of input and output channels
        '''
        super().__init__()
        self.nc = nc
        self.norm = nn.GroupNorm(8, self.nc)
        self.attn = MHA(self.nc, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H*W).permute(0, 2, 1)
        #h = rearrange(self.norm(x), 'B C H W -> B (H W) C')
        h = self.attn(h)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        #h = rearrange(h, 'B (H W) C -> B C H W', H=H, W=W)
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
        #x = rearrange(x, 'B C H W -> B (H W) C')
        x = x.reshape(B, C, H*W).permute(0, 2, 1).contiguous()
        x = self.attn1(self.norm1(x)) + x
        if context is not None:
            x = self.attn2(self.norm2(x), context) + x
        x = self.ff(self.norm3(x)) + x
        #x = rearrange(x, 'B (H W) C -> B C H W', H=H, W=W)
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        return x + skip
