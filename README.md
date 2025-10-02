# EAX

<a href="https://arxiv.org/abs/2509.19770">
  <img src="https://img.shields.io/badge/EAX-Paper-blue"></a>
<a href="https://huggingface.co/collections/double7/enanchored-x2x-6830338f017061c30226107d">
  <img src="https://img.shields.io/badge/EAX-Hugging Face-brightgreen"></a>
<a href="LICENSE">
  <img src="https://img.shields.io/badge/License-MIT-yellow"></a>


Work in progress...

## Getting Started

### Installation

> [!IMPORTANT]
> Installation is mandatory.

#### Install from Source

```bash
git clone https://github.com/NJUNLP/EAX.git
cd EAX
pip install -e ".[infer]" --no-build-isolation
```

Extra dependencies available:
- `infer`: install vllm for sampling.
- `eval`: comet, sacrebleu and bleurt for evaluation. Also, bleurt is required for Reward Modeling.

### x2x Optimization Pipeline


The pipeline includes the following steps:
1. [Supervised Fine-tuning](recipes/sft.md): setup the translation model with supervised data.
2. [Reward Modeling](recipes/rm.md): build translation evaluation capabilities for the SFT model through Reward Modeling.
3. [x2x Optimization](recipes/xpo.md): optimize x2x translation with English-Anchored Generation and Evaluation.

## Citation 

```bibtex
@misc{yang2025enanchoredx2xenglishanchoredoptimizationmanytomany,
      title={EnAnchored-X2X: English-Anchored Optimization for Many-to-Many Translation}, 
      author={Sen Yang and Yu Bao and Yu Lu and Jiajun Chen and Shujian Huang and Shanbo Cheng},
      year={2025},
      eprint={2509.19770},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.19770}, 
}
```