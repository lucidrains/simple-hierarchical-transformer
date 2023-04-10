## Simple Hierarchical Transformer (wip)

Experiments around a simple idea for inducing multiple hierarchical predictive coding models within a GPT. It is so simple, it may not work. But then again, deep learning progress is built on the bedrocks of simple ideas. Worth a shot.

So far, the idea has passed the litmus test from a research friend. Will bring it to completion in the next week or so. If it does not work out, I'll leave the negative experimental results as well as the repository around, and maybe some PhD student can build upon it.

## Appreciation

- <a href="https://stability.ai/">StabilityAI</a> for the sponsorship to carry out this independent research

- <a href="https://huggingface.co/">🤗 Huggingface</a> for their accelerate library

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
