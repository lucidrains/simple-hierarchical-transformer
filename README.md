## Simple Hierarchical Transformer

Experiments around a simple idea for inducing multiple hierarchical predictive coding models within a GPT. It is so simple, it may not work. But then again, deep learning progress is built on the bedrocks of simple ideas. Worth a shot.

So far, the idea has passed the litmus test from a research friend. Will bring it to completion in the next week or so. If it does not work out, I'll leave the negative experimental results as well as the repository around, and maybe some PhD student can build upon it.

<a href="https://api.wandb.ai/links/lucidrains/w8vdkz75">Ongoing experiments</a>

Update: I think it is working 🤞 

## Appreciation

- <a href="https://stability.ai/">StabilityAI</a> for the sponsorship to carry out this independent research

- <a href="https://huggingface.co/">🤗 Huggingface</a> for their accelerate library

## Install

```
$ pip install simple-hierarchical-transformer
```

## Usage

Three hierarchies, all servicing predicting the next token

```python
import torch
from simple_hierarchical_transformer import HierarchicalTransformer

model = HierarchicalTransformer(
    num_tokens = 20000,                # number of tokens
    dim = 512,                         # model dimensions
    depth = 6,                         # depth
    dim_head = 64,                     # dimension per attention head
    heads = 8,                         # attention heads
    seq_len = 2048,                    # sequence lengths
    hierarchies = (1, 2, 8),           # hierarchies - here we have 1x (like in a regular transformer), then 2x and 8x compressed hierarchical tokens that undergo their own transformer blocks. information is pooled into one hierarchy at each layer
    window_sizes = (32, 64, None)      # local attention window sizes - the idea is that the higher hierarchies can pass distant information to the local one. None stands for full receptive field
)

ids = torch.randint(0, 20000, (1, 2048))

loss, _ = model(ids, return_loss = True)
loss.backward()

# after much training

logits = model(ids)
```

By not specifying `hierarchies` and `window_sizes`, you basically default to a regular autoregressive transformer with attention across full sequence length.

```python

# non-hierarchical transformer

model = HierarchicalTransformer(
    num_tokens = 20000,
    dim = 512,
    depth = 8,
    dim_head = 64,
    heads = 8,
    seq_len = 2048,
    hierarchies = 1,        # implied 1 if not set
    window_sizes = None     # implied None (full sequence length) if not set
)

```

Now something more complex and appropriate. Experiments show that as you compress up the hierarchies, you need greater model dimensions for appropriate capacity.

```python
model = HierarchicalTransformer(
    num_tokens = 256,
    dim = (128, 256, 512),
    depth = 8,
    seq_len = SEQ_LEN,
    use_flash_attn = True,
    ff_mult = (1, 2, 4),
    dim_head = (16, 32, 64),
    heads = (2, 4, 8),
    hierarchies = (1, 2, 4),
    window_sizes = (16, 32, 64)
).cuda()

# hierarchies
# 1x - dim 128 - attention (2 heads, 16 dim, receptive field 16)
# 2x - dim 256 - attention (4 heads, 32 dim, receptive field 32)
# 4x - dim 512 - attention (8 heads, 64 dim, receptive field 64)

```

## Todo

- [x] branch out to two parallel paths, one for hierarchical tokens, other for plain fine tokens.
- [x] show that local attention in fine + hierarchical tokens can come close to full attention baseline
- [x] simple dsconv seems enough to merge for 1 hierarchy
- [x] auto-set window size to be half of max sequence length for fine and all hierarchies
- [x] figure out effects of just pooling all fine + hierarchical tokens before cross entropy loss - not much of a difference
- [x] complete ability to add any number of hierarchies, and designate which hierarchy will pool the information from the others for prediction
- [x] fully customizable dimensions across hierarchies, as higher hierarchies require greater model dimensions

- [ ] try a few types of attention across hierarchies. full self attention, directional, or even token shift and feedforward
- [ ] play around with an autoregressive loss on the hierarchy tokens, can try with random projections + vq, as was done in universal speech model paper from brain - also try prophet loss
- [ ] allow for repeating hierarchy tokens for fine tokens in the future, as position may matter less as one goes up the hierarchy. but not a priority, get things working first
- [ ] build out simple local attention block, for use across all hierarchies
- [ ] add flash attention to local attention library
- [ ] figure out if attention can be shared across hierarchies


## Citations

Closest idea would be <a href="https://arxiv.org/abs/2110.13711">hourglass transformers</a>.

And my renewed interest in hierarchical approaches came from reading <a href="https://www.nature.com/articles/s41562-022-01516-2">this</a>.

```bibtex
@article{Nawrot2021HierarchicalTA,
    title   = {Hierarchical Transformers Are More Efficient Language Models},
    author  = {Piotr Nawrot and Szymon Tworkowski and Michal Tyrolski and Lukasz Kaiser and Yuhuai Wu and Christian Szegedy and Henryk Michalewski},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2110.13711}
}
```

```bibtex
@inproceedings{dao2022flashattention,
    title   = {Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
    author  = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
    booktitle = {Advances in Neural Information Processing Systems},
    year    = {2022}
}
```

```bibtex
@article{Yan2020ProphetNetPF,
    title   = {ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training},
    author  = {Yu Yan and Weizhen Qi and Yeyun Gong and Dayiheng Liu and Nan Duan and Jiusheng Chen and Ruofei Zhang and Ming Zhou},
    journal = {ArXiv},
    year    = {2020},
    volume  = {abs/2001.04063}
}
```

```bibtex
@misc{su2021roformer,
    title   = {RoFormer: Enhanced Transformer with Rotary Position Embedding},
    author  = {Jianlin Su and Yu Lu and Shengfeng Pan and Bo Wen and Yunfeng Liu},
    year    = {2021},
    eprint  = {2104.09864},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```

```bibtex
@inproceedings{Sun2022ALT,
    title     = {A Length-Extrapolatable Transformer},
    author    = {Yutao Sun and Li Dong and Barun Patra and Shuming Ma and Shaohan Huang and Alon Benhaim and Vishrav Chaudhary and Xia Song and Furu Wei},
    year      = {2022}
}
```

```bibtex
@software{peng_bo_2021_5196578,
    author    = {PENG Bo},
    title     = {BlinkDL/RWKV-LM: 0.01},
    month     = {aug},
    year      = {2021},
    publisher = {Zenodo},
    version   = {0.01},
    doi       = {10.5281/zenodo.5196578},
    url       = {https://doi.org/10.5281/zenodo.5196578}
}
```
