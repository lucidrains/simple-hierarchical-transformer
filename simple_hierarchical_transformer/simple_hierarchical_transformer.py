import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def default(*vals):
    for val in vals:
        if exists(val):
            return val
    return None

mlist = nn.ModuleList

# classes

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        dim_inner = dim_head * heads

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(self, x):
        n = x.shape[-2]
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        causal_mask = torch.ones((n, n), device = x.device, dtype = torch.bool).triu(1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class HierarchicalTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        seq_len = 2048,
        dim_head = 64,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        self.seq_len = seq_len

        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = mlist([])

        for _ in range(depth):
            self.layers.append(mlist([
                mlist([
                    RMSNorm(dim),
                    Attention(dim = dim, dim_head = dim_head, heads = heads),
                ]),
                mlist([
                    RMSNorm(dim),
                    FeedForward(dim = dim, mult = ff_mult)
                ])
            ]))

        self.to_logits = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, num_tokens, bias = False)
        )

    def forward(self, x):
        x = self.token_emb(x)

        for (attn_prenorm, attn), (ff_prenorm, ff) in self.layers:
            x = attn(attn_prenorm(x)) + x
            x = ff(ff_prenorm(x)) + x

        return self.to_logits(x)
