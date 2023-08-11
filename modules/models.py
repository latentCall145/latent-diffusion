from modules.layers import ResBlock, Downsample, Upsample, Attn2d, TransformerBlock
from transformers import CLIPTokenizerFast, CLIPTextModel
from torchvision.models import vgg16, VGG16_Weights
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

class VAEEncoder(nn.Module):
    def __init__(self,
            nc: int,
            ch_mults: tuple,
            nlayers_per_res: int,
            nz: int
            ):
        '''
        nc: base number of channels
        ch_mults: channel multiplier per resolution
        nz: number of latent output channels
        '''
        super().__init__()
        # put input args as class fields
        locals_ = locals().copy()
        locals_.pop('self')
        for k, v in locals_.items():
            setattr(self, k, v)

        layers = [nn.Conv2d(3, nc, 3, padding=1)]
        out_c = nc

        for layer_idx, ch_mult in enumerate(ch_mults):
            in_c, out_c = out_c, nc * ch_mult
            layers += [ResBlock(in_c, out_c)] + [ResBlock(out_c, out_c) for _ in range(nlayers_per_res-1)]
            if layer_idx != len(ch_mults) - 1:
                layers.append(Downsample(out_c))

        layers += [ 
            # mid
            ResBlock(out_c, out_c),
            Attn2d(out_c),
            ResBlock(out_c, out_c),

            # out
            nn.GroupNorm(8, out_c),
            nn.SiLU(),
            nn.Conv2d(out_c, 2*nz, 3, padding=1),
        ]

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.model(x) # nz*2 for the mean, stdev
        return x

class VAEDecoder(nn.Module):
    def __init__(self,
            nc: int,
            ch_mults: tuple,
            nlayers_per_res: int,
            nz: int
            ):
        '''
        nc: base number of channels
        ch_mults: channel multiplier per resolution
        nz: number of latent output channels
        '''
        super().__init__()
        # put input args as class fields
        locals_ = locals().copy()
        locals_.pop('self')
        for k, v in locals_.items():
            setattr(self, k, v)

        out_c = self.nc * self.ch_mults[-1] 
        layers = [
            nn.Conv2d(nz, out_c, 3, padding=1),
            ResBlock(out_c, out_c),
            Attn2d(out_c),
            ResBlock(out_c, out_c),
        ]

        for layer_idx, ch_mult in enumerate(reversed(ch_mults)):
            in_c, out_c = out_c, self.nc * ch_mult
            layers += [ResBlock(in_c, out_c)] + [ResBlock(out_c, out_c) for _ in range(nlayers_per_res-1)]
            if layer_idx != 0:
                layers.append(Upsample(out_c))

        layers += [
            nn.GroupNorm(8, out_c),
            nn.SiLU(),
            nn.Conv2d(out_c, 3, 3, padding=1)
        ]

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
class VAE(nn.Module):
    def __init__(
        self,
        nc: int = 128,
        ch_mults: list = [1, 2, 2, 4],
        nlayers_per_res: int = 2,
        nz: int = 4
    ):
        super().__init__()
        # put input args as class fields
        locals_ = locals().copy()
        locals_.pop('self')
        for k, v in locals_.items():
            setattr(self, k, v)

        self.encoder = VAEEncoder(nc=self.nc, ch_mults=self.ch_mults, nlayers_per_res=self.nlayers_per_res, nz=self.nz)
        self.decoder = VAEDecoder(nc=self.nc, ch_mults=self.ch_mults, nlayers_per_res=self.nlayers_per_res, nz=self.nz)

    def forward(self, x):
        z_params = self.encoder(x)
        mean, log_var = torch.split(z_params, self.nz, dim=1)
        z = self.sample(mean, log_var)
        out = self.decoder(z)
        return out, mean, log_var

    def sample(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(log_var)
        return mean + eps * std

class Discriminator(nn.Module):
    '''
    just a normal PatchGAN discriminator model - copied from https://github.com/CompVis/taming-transformers/blob/master/taming/modules/discriminator/model.py
    '''
    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 3):
        '''
        input_nc: the number of channels in input images
        ndf: the number of filters in the last conv layer
        n_layers: the number of conv layers in the discriminator
        norm_layer: normalization layer
        '''
        super().__init__()
        use_bias = False # because batch norm after conv nullifies bias

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakySiLU(0.2)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                nn.GroupNorm(8, ndf * nf_mult),
                nn.LeakySiLU(0.2)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            nn.GroupNorm(8, ndf * nf_mult),
            nn.LeakySiLU(0.2)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, x):
        return self.main(x)

class FrozenCLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=64):
        super().__init__()
        self.tokenizer = CLIPTokenizerFast.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version).to(device)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

class TimestepEmbedSequential(nn.Sequential):
    '''
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    '''
    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, (ResBlock)):
                x = layer(x, emb)
            elif isinstance(layer, TransformerBlock):
                x = layer(x, context)
            else:
                x = layer(x)
        return x

class UNet(nn.Module):
    def __init__(self,
        in_c: int = 4,
        nc: int = 256,
        ch_mults: list = [1, 2, 4],
        attn_resolutions: list = [0, 1, 2],
        nlayers_per_res: int = 2,
        context_dim: int = 768,
    ):
        super().__init__()
        # put input args as class fields
        locals_ = locals().copy()
        locals_.pop('self')
        for k, v in locals_.items():
            setattr(self, k, v)

        temb_c = nc * 4
        self.time_embed = nn.Embedding(10000, temb_c)

        # adding downsampling blocks
        out_c = nc
        self.downs = nn.ModuleList([TimestepEmbedSequential(nn.Conv2d(in_c, nc, 3, padding=1))])
        down_out_cs = [nc] # output channel counts of the downsampling activation stack
        for block_idx, ch_mult in enumerate(ch_mults):
            for n_idx in range(nlayers_per_res):
                in_c, out_c = out_c, nc * ch_mult

                block = [ResBlock(in_c, out_c, temb_c)] + [TransformerBlock(out_c, context_dim) for _ in range(attn_resolutions[block_idx])]

                block = TimestepEmbedSequential(*block)
                self.downs.append(block)
                down_out_cs.append(out_c)

            if block_idx != len(ch_mults) - 1:
                self.downs.append(TimestepEmbedSequential(Downsample(out_c)))
                down_out_cs.append(out_c)

        # middle block
        n_mid_transformer_blocks = max(1, attn_resolutions[-1])
        self.mid_block = TimestepEmbedSequential(
                ResBlock(out_c, out_c, temb_c),
                *[TransformerBlock(out_c, context_dim) for _ in range(n_mid_transformer_blocks)],
                ResBlock(out_c, out_c, temb_c),
            )

        # adding upsampling blocks
        self.ups = nn.ModuleList([])
        for block_idx, ch_mult in enumerate(reversed(ch_mults)):
            for n_idx in range(nlayers_per_res+1):
                down_c = down_out_cs.pop()
                in_c, out_c = out_c + down_c, nc * ch_mult

                block = [ResBlock(in_c, out_c, temb_c)] + [TransformerBlock(out_c, context_dim) for _ in range(attn_resolutions[-block_idx-1])]

                if n_idx == nlayers_per_res and block_idx != len(ch_mults)-1:
                    block.append(Upsample(out_c))

                block = TimestepEmbedSequential(*block)
                self.ups.append(block)

        self.out = nn.Sequential(
            nn.GroupNorm(8, nc),
            nn.SiLU(),
            nn.Conv2d(nc, self.in_c, 3, padding=1)
        )

    def forward(self, x, timesteps, context=None):
        temb = self.time_embed(timesteps)
        downs = []
        for block in self.downs:
            x = block(x, temb, context)
            downs.append(x)
        x = self.mid_block(x, temb, context)
        for block in self.ups:
            x = torch.cat([x, downs.pop()], dim=1)
            x = block(x, temb, context)
        x = self.out(x)

        return x

UNetSD = lambda: UNet(in_c=4, nc=320, ch_mults=[1,2,4,4], attn_resolutions=[1,1,1,0], nlayers_per_res=2, context_dim=768)
UNetSDXL = lambda: UNet(in_c=4, nc=320, ch_mults=[1,2,4], attn_resolutions=[0,2,10], nlayers_per_res=2, context_dim=2048)
