# ⚓EAX

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

### Evaluation on FLORES

Setup the flores dataset:
```bash
wget https://tinyurl.com/flores200dataset -O flores200dataset.tar.gz
tar -xzvf flores200dataset.tar.gz
ls flores200_dataset
```

Evaluate the model on flores dataset and log the results to wandb:
```bash
python3 eval/run_eval_flores.py \
  --model_path path/to/model \
  --model_name model_name_for_logging \
  --test_data_path flores200_dataset \
  --split devtest \
  --metrics ["bleurt","sacrebleu","comet"] \
  --bleurt_path BLEURT-20 \
  --comet_path wmt22-comet-da/checkpoints/model.ckpt \
  --log_to_wandb True
```

> [!TIP]
> set `--log_to_wandb False` if wandb is not available and the results are logged to console.

> [!IMPORTANT]
> Do not evaluate our models on flores `dev` split as it is included in the [Towerblocks](https://huggingface.co/datasets/Unbabel/TowerBlocks-v0.1) dataset for training.

### Evaluation on custom dataset

You can evaluate the model on custom dataset by preparing the inference data in the following format:
```json
[
  {
    "src_lang": "en",
    "trg_lang": "zh",
    "src_text": "\"We now have 4-month-old mice that are non-diabetic that used to be diabetic,\" he added.",
    "trg_text": "他补充道：“我们现在有 4 个月大没有糖尿病的老鼠，但它们曾经得过该病。”",
  },
  ...
]
```

Run the evaluation:
```bash
python3 eval/run_eval.py \
  --model_path path/to/model \
  --infer_data_path path/to/infer_data.json \
  --metrics ["bleurt","sacrebleu","comet"] \
  --bleurt_path BLEURT-20 \
  --comet_path wmt22-comet-da/checkpoints/model.ckpt \
  --log_to_wandb True \
  --config '{"model_name": "qwen7b_eax"}' # any info that you want to log to wandb
```


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
