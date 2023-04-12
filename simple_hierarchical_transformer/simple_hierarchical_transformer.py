import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from simple_hierarchical_transformer.attention import Attend

from local_attention import LocalMHA

# constants

mlist = nn.ModuleList

Linear = partial(nn.Linear, bias = False)

LocalMHA = partial(LocalMHA, causal = True, prenorm = True)

# helper functions

def exists(val):
    return val is not None

def is_power_of_two(n):
    return math.log2(n).is_integer()

def all_unique(arr):
    return len(set(arr)) == len(arr)

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

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
    seq_len = t.shape[-2]

    pos = pos[..., -seq_len:, :]
    if not isinstance(scale, (int, float)):
        scale = scale[..., -seq_len:, :]

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
        *,
        dim,
        num_tokens,
        compress_factor = 2,
        expansion_factor = 4,
        dim_head = 64,
        heads = 8,
        ignore_index = 0
    ):
        super().__init__()
        assert compress_factor > 0 and is_power_of_two(compress_factor)

        self.no_compress = compress_factor == 1
        self.compress_factor = compress_factor

        if self.no_compress:
            self.compress_fn = nn.Identity()
            return

        dim_inner = int(dim * expansion_factor)

        self.compress_fn = nn.Sequential(
            Rearrange('b n d -> b d n'),
            CausalConv(dim, dim_inner, compress_factor),
            nn.SiLU(),
            nn.Conv1d(dim_inner, dim, 1),
            Rearrange('b d n -> b n d')
        )

        self.to_recon = Linear(dim, compress_factor * num_tokens)
        self.ignore_index = ignore_index

    def recon(self, h, ids):

        if self.no_compress:
            return torch.zeros((), device = h.device).requires_grad_()

        c = self.compress_factor
        seq_len = ids.shape[-1]

        recon_logits = self.to_recon(h)
        recon_logits = rearrange(recon_logits, 'b n (c d) -> (b c) d n', c = c)

        recon_ids = F.pad(ids, (c - 1, 0), value = 0)
        recon_ids = tuple(recon_ids[:, i:(seq_len + i)] for i in range(c))
        recon_ids = torch.stack(recon_ids, dim = 1)
        recon_ids = rearrange(recon_ids, 'b c n -> (b c) n')

        recon_loss = F.cross_entropy(recon_logits, recon_ids, ignore_index = self.ignore_index)
        return recon_loss

    def forward(self, x):
        return self.compress_fn(x)

class HierarchicalMerge(nn.Module):
    def __init__(
        self,
        dim,
        num_hierarchies = 2
    ):
        super().__init__()
        self.norms = mlist([RMSNorm(dim) for _ in range(num_hierarchies)])

        # simple dsconv for now

        self.num_hierarchies = num_hierarchies
        self.conv = nn.Conv1d(dim, dim, num_hierarchies, stride = num_hierarchies, groups = dim)

    def forward(self, tokens):
        """
        einops notations:
        b - batch
        h - hierarchies
        n - sequence length
        d - dimension
        """
        nh = self.num_hierarchies

        tokens = [norm(h) for norm, h in zip(self.norms, tokens)]

        x = rearrange(tokens, 'h b n d -> b d (n h)')
        x = self.conv(x)
        x = rearrange(x, 'b d n -> b n d')
        return x

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
        return self.net(x)

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
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.attend = Attend(causal = True, use_flash_attn = use_flash_attn)

        self.to_qkv = Linear(dim, dim_inner * 3)
        self.to_out = Linear(dim_inner, dim)

    def forward(self, x):
        n = x.shape[-2]
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        rotary_emb = self.rotary_emb(n)
        q, k = apply_rotary_pos_emb_qk(rotary_emb, q, k)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class HierarchicalAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        window_size = None,
        compress_factor = 1
    ):
        super().__init__()
        assert is_power_of_two(compress_factor)
        self.compress_factor = compress_factor
        self.no_compress = compress_factor == 1

        attn_klass = Attention

        if exists(window_size):
            attn_klass = partial(LocalMHA, window_size = window_size)

        self.attn = attn_klass(dim = dim, dim_head = dim_head, heads = heads)

    def forward(self, x):
        c = self.compress_factor
        x, orig_seq_len = pad_seq_to_multiple(x, c)

        # hierarchical attention is performed with a simple axial attention

        # this, and using a convolution for compressing at the beginning
        # is one of the improvements on top of hourglass transformer
        # the downside is that the savings are only O(c) instead of O(c ** 2)
        # but this should provide better learning per-token

        if not self.no_compress:
            x = rearrange(x, 'b (n c) d -> (b c) n d', c = c)

        x = self.attn(x)

        if not self.no_compress:
            x = rearrange(x, '(b c) n d -> b (n c) d', c = c)

        return x[:, :orig_seq_len]

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
        hierarchies = 1,
        predict_hierarchy = None,
        window_sizes = None,
        ignore_index = 0,
        recon_loss_weight = 0.1,
        use_flash_attn = False
    ):
        super().__init__()
        self.seq_len = seq_len

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.post_token_emb_norm = RMSNorm(dim)

        hierarchies = tuple(sorted(cast_tuple(hierarchies)))

        assert all_unique(hierarchies), 'hierarchies compression factors must be all unique integers'
        assert all([*map(is_power_of_two, hierarchies)]), 'only powers of two allowed for hierarchies'

        num_hierarchies = len(hierarchies)
        predict_hierarchy = default(predict_hierarchy, min(hierarchies))

        self.predict_hierarchy_index = hierarchies.index(predict_hierarchy)

        window_sizes = cast_tuple(window_sizes, num_hierarchies)
        assert len(window_sizes) == len(hierarchies)

        # hierarchy compressions - 1x just uses the base token_emb weights

        self.compressors = mlist([])

        for hierarchy in hierarchies:
            self.compressors.append(Compress(
                dim = dim,
                num_tokens = num_tokens,
                compress_factor = hierarchy,
            ))

        # layers

        self.layers = mlist([])
        self.hierarchical_merges = mlist([])

        local_attn = partial(LocalMHA, causal = True, prenorm = True)

        for _ in range(depth):
            hierarchical_layer = mlist([])

            # add a transformer block for each layer in the hierarchy

            for hierarchy, window_size in zip(hierarchies, window_sizes):

                # make sure the window size never exceeds the effective sequence length

                effective_seq_len = seq_len // hierarchy

                if exists(window_size) and window_size > effective_seq_len:
                    print(f'window size for hierarchy {hierarchy}x is greater than effective sequence length - setting window size to None (which would use normal full attention)')
                    window_size = None

                # add attention and feedforward

                hierarchical_layer.append(mlist([
                    HierarchicalAttention(
                        dim = dim,
                        dim_head = dim_head,
                        heads = heads,
                        window_size = window_size,
                        compress_factor = hierarchy
                    ),
                    FeedForward(dim = dim, mult = ff_mult)
                ]))

            self.layers.append(hierarchical_layer)

            # for merging the information across hierarchies
            # for now, only one direction, from all hierarchies to the hierarchy that is being used to make predictions on, set by predict_hierarchy_index above

            merge = HierarchicalMerge(dim = dim, num_hierarchies = num_hierarchies)

            self.hierarchical_merges.append(merge)

        # final post-transformer norms, for all hierarchies

        self.norms = mlist([RMSNorm(dim) for _ in range(num_hierarchies)])

        # to logit, for hierarchy set at predict_hierarchy_index

        self.to_logits = Linear(dim, num_tokens)

        # training related loss parameters

        self.ignore_index = ignore_index
        self.recon_loss_weight = recon_loss_weight

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
        return_loss = False,
        ablate_hierarchical_merge = False
    ):
        """
        einops notation:

        b - batch
        n - sequence length
        c - compression factor
        d - dimension
        """

        # if training, predict next token in sequence

        if return_loss:
            ids, labels = ids[:, :-1], ids[:, 1:]

        # assert seq len

        assert ids.shape[-1] <= self.seq_len

        # get token embeddings, and pad to multiple of compression factor

        x = self.token_emb(ids)

        # post embedding norm

        x = self.post_token_emb_norm(x)

        # for every hierarchy, compress token embeddings appropriately to the hierarchical embeddings

        tokens = []

        for compress in self.compressors:
            tokens.append(compress(x))

        # layers

        for layer, merge in zip(self.layers, self.hierarchical_merges):

            next_tokens = []

            for (attn, ff), h in zip(layer, tokens):
                h = attn(h) + h
                h = ff(token_shift(h)) + h

                next_tokens.append(h)

            tokens = next_tokens

            # pool the information all hierarchies
            # and then update the tokens that will be used to make the final autoregressive prediction

            if ablate_hierarchical_merge:
                continue

            pooled = merge(tokens)
            predict_tokens = tokens[self.predict_hierarchy_index]
            predict_tokens = predict_tokens + pooled
            tokens[self.predict_hierarchy_index] = predict_tokens

        # final norm and logits

        tokens = [norm(t) for norm, t in zip(self.norms, tokens)]

        # select the hierarchical embeddings that will be doing the predicting

        predict_embed = tokens[self.predict_hierarchy_index]

        # logits for predicting next token

        logits = self.to_logits(predict_embed)

        if not return_loss:
            return logits

        # autoregressive loss (predictive coding)

        logits = rearrange(logits, 'b n c -> b c n')
        ce_loss = F.cross_entropy(logits, labels, ignore_index = self.ignore_index)

        # reconstruction losses for hierarchy tokens -> may remove if see no benefit, which seems to be leaning that way

        recon_losses = 0

        for compress, t in zip(self.compressors, tokens):
            recon_loss = compress.recon(t, ids)
            recon_losses = recon_losses + recon_loss

        # total loss

        total_loss = ce_loss + recon_loss * self.recon_loss_weight

        return total_loss, (ce_loss, recon_loss)
