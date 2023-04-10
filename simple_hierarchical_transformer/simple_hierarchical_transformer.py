from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange

from simple_hierarchical_transformer.attention import Attend

# constants

mlist = nn.ModuleList

Linear = partial(nn.Linear, bias = False)

# helper functions

def exists(val):
    return val is not None

def default(*vals):
    for val in vals:
        if exists(val):
            return val
    return None

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# sampling helpers

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, -torch.finfo(logits.dtype).max)
    probs.scatter_(1, ind, val)
    return probs

# rotary positional embedding w/ xpos
# https://arxiv.org/abs/2104.09864
# https://arxiv.org/abs/2212.10554v1

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        scale_base = 512,
        use_xpos = True
    ):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.use_xpos = use_xpos
        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale)

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, seq_len):
        device = self.device
        t = torch.arange(seq_len, device = device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        if not self.use_xpos:
            return freqs, torch.ones(1, device = device)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(pos, t, scale = 1.):
    return (t * pos.cos() * scale) + (rotate_half(t) * pos.sin() * scale)

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
        RMSNorm(dim),
        Linear(dim, dim_inner),
        nn.GELU(),
        Linear(dim_inner, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        use_flash_attn = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        dim_inner = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(causal = True, use_flash_attn = use_flash_attn)

        self.to_qkv = Linear(dim, dim_inner * 3)
        self.to_out = Linear(dim_inner, dim)

    def forward(
        self,
        x,
        rotary_emb = None
    ):
        n = x.shape[-2]
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        if exists(rotary_emb):
            freqs, scale = rotary_emb
            q = apply_rotary_pos_emb(freqs, q, scale)
            k = apply_rotary_pos_emb(freqs, k, scale ** -1)

        out = self.attend(q, k, v)

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
        ff_mult = 4,
        use_flash_attn = False,
        ignore_index = 0
    ):
        super().__init__()
        self.seq_len = seq_len
        self.ignore_index = ignore_index

        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = mlist([])

        self.rotary_emb = RotaryEmbedding(dim_head)

        for _ in range(depth):
            self.layers.append(mlist([
                Attention(dim = dim, dim_head = dim_head, heads = heads, use_flash_attn = use_flash_attn),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.to_logits = nn.Sequential(
            RMSNorm(dim),
            Linear(dim, num_tokens)
        )

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        prompt,
        seq_len,
        temperature = 1.0,
        filter_thres = 0.9,
        **kwargs
    ):
        b, t, device = *prompt.shape, prompt.device

        out = prompt

        for _ in range(seq_len):
            logits = self.forward(out[:, -self.seq_len:], **kwargs)[:, -1]
            filtered_logits = top_k(logits, thres = filter_thres)
            sample = gumbel_sample(filtered_logits, temperature = temperature)
            sample = rearrange(sample, 'b -> b 1')
            out = torch.cat((out, sample), dim = -1)

        return out[:, t:]

    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(
        self,
        x,
        return_loss = False
    ):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        n = x.shape[-1]

        x = self.token_emb(x)

        pos_emb = self.rotary_emb(n)

        for attn, ff in self.layers:
            x = attn(x, rotary_emb = pos_emb) + x
            x = ff(x) + x

        logits = self.to_logits(x)

        if not return_loss:
            return logits

        logits = rearrange(logits, 'b n c -> b c n')
        return F.cross_entropy(logits, labels, ignore_index = self.ignore_index)
