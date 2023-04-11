import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from simple_hierarchical_transformer.attention import Attend

# constants

mlist = nn.ModuleList

Linear = partial(nn.Linear, bias = False)

# helper functions

def exists(val):
    return val is not None

def is_power_of_two(n):
    return math.log2(n).is_integer()

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

def apply_rotary_pos_emb_qk(rotary_emb, q, k):
    freqs, scale = rotary_emb
    q = apply_rotary_pos_emb(freqs, q, scale)
    k = apply_rotary_pos_emb(freqs, k, scale ** -1)
    return q, k

# token shift, from Peng et al of RWKV

def token_shift(t):
    t, t_shift = t.chunk(2, dim = -1)
    t_shift = F.pad(t_shift, (0, 0, 1, -1))
    return torch.cat((t, t_shift), dim = -1)

# hierarchy related classes

def pad_seq_to_multiple(t, mult):
    seq_len = t.shape[-2]
    next_seq_len_mult = math.ceil(seq_len / mult) * mult
    remainder = next_seq_len_mult - seq_len

    if remainder == 0:
        return t, seq_len

    t = F.pad(t, (0, 0, 0, remainder), value = 0.)
    return t, seq_len

class CausalConv(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size
    ):
        super().__init__()
        self.causal_padding = kernel_size - 1
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size)

    def forward(self, x):
        x = F.pad(x, (self.causal_padding, 0))
        return self.conv(x)

class Compress(nn.Module):
    def __init__(
        self,
        dim,
        compress_factor = 2,
        expansion_factor = 4,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        assert compress_factor > 1 and is_power_of_two(compress_factor)

        self.compress_factor = compress_factor
        dim_inner = int(dim * expansion_factor)

        self.compress_fn = nn.Sequential(
            Rearrange('b n d -> b d n'),
            CausalConv(dim, dim_inner, compress_factor),
            nn.SiLU(),
            nn.Conv1d(dim_inner, dim, 1),
            Rearrange('b d n -> b n d')
        )

    def forward(self, x):
        batch, factor = x.shape[0], self.compress_factor

        pooled = self.compress_fn(x)

        x = rearrange(x, 'b n d -> b d n 1')
        x = F.pad(x, (0, 0, factor - 1, 0), value = 0.)
        unfolded = F.unfold(x, (factor, 1))
        unfolded = rearrange(unfolded, 'b (d c) n -> (b n) c d', c = factor)

        return pooled, unfolded

# classes

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        dim_inner = int(dim * mult)

        self.net = nn.Sequential(
            RMSNorm(dim),
            Linear(dim, dim_inner),
            nn.GELU(),
            Linear(dim_inner, dim)
        )

    def forward(self, x):
        x = self.net(x)
        return x

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
            q, k = apply_rotary_pos_emb_qk(rotary_emb, q, k)

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
        ignore_index = 0,
        compress_factor = 1,
        recon_loss_weight = 0.1
    ):
        super().__init__()
        assert is_power_of_two(compress_factor)

        self.seq_len = seq_len
        self.ignore_index = ignore_index

        self.token_emb = nn.Embedding(num_tokens, dim)

        should_compress = compress_factor > 1

        self.compress_factor = compress_factor
        self.compress = None

        if should_compress:
            self.compress = Compress(
                dim,
                compress_factor = compress_factor
            )

        self.recon_loss_weight = recon_loss_weight

        self.layers = mlist([])

        self.rotary_emb = RotaryEmbedding(dim_head)

        for _ in range(depth):
            self.layers.append(mlist([
                Attention(dim = dim, dim_head = dim_head, heads = heads, use_flash_attn = use_flash_attn),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = RMSNorm(dim)

        self.to_logits = Linear(dim, num_tokens)
        self.to_recon = Linear(dim, compress_factor * num_tokens) if should_compress else None

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
        ids,
        return_loss = False
    ):
        """
        einops notation:

        b - batch
        n - sequence length
        c - compression factor
        d - dimension
        """

        # whether to compress or not

        c = self.compress_factor
        should_compress = c > 1

        # if training, predict next token in sequence

        if return_loss:
            ids, labels = ids[:, :-1], ids[:, 1:]

        # get token embeddings, and pad to multiple of compression factor

        x = self.token_emb(ids)
        x, orig_seq_len = pad_seq_to_multiple(x, c)

        # compress to hierarchical tokens from the beginning

        h = x
        if exists(self.compress):
            h, uncompressed = self.compress(x)

        # rotary positional embeddings

        pos_emb = self.rotary_emb(h.shape[-2] // c)

        # layers

        for attn, ff in self.layers:
            if should_compress:
                h = rearrange(h, 'b (n c) d -> (b c) n d', c = c)

            h = attn(h, rotary_emb = pos_emb) + h

            if should_compress:
                h = rearrange(h, '(b c) n d -> b (n c) d', c = c)

            h = ff(token_shift(h)) + h

        # get back the original sequence length

        h = h[:, :orig_seq_len]

        # final norm and logits

        h = self.norm(h)

        logits = self.to_logits(h)

        if not return_loss:
            return logits

        ce = partial(F.cross_entropy, ignore_index = self.ignore_index)

        # reconstruction losses for hierarchy tokens -> may remove if see no benefit, which seems to be leaning that way

        recon_loss = torch.zeros((), device = self.device)
        if should_compress:
            recon_logits = self.to_recon(x)
            recon_logits = rearrange(recon_logits, 'b n (c d) -> (b c) d n', c = c)

            recon_ids = F.pad(ids, (c - 1, 0), value = 0)
            recon_ids = tuple(recon_ids[:, i:(orig_seq_len + i)] for i in range(c))
            recon_ids = torch.stack(recon_ids, dim = 1)
            recon_ids = rearrange(recon_ids, 'b c n -> (b c) n')
            recon_loss = ce(recon_logits, recon_ids)

        logits = rearrange(logits, 'b n c -> b c n')
        ce_loss = ce(logits, labels)

        total_loss = ce_loss + recon_loss * self.recon_loss_weight
        return total_loss, (ce_loss, recon_loss)
