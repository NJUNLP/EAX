
# SFT Recipe for Base Model

The purpose of supervised fine-tuning is to activate the English-centric translation capabilities of base models.
We use approximately 150k translation data samples from [TowerBlocks](https://huggingface.co/datasets/Unbabel/TowerBlocks-v0.1).


## Data Preprocessing

We have prepared the translation task data extracted from TowerBlocks. Run the following command to download it:
```bash
hf download double7/TowerBlocks-MT --repo-type dataset --local-dir TowerBlocks-MT
```


Run the command below to add translation instructions to the translation pairs for supervised fine-tuning:
```bash
python3 scripts/prepare_sft_instruction_data.py --data_path  TowerBlocks-MT/data/train.parquet --output_path sft_towerblocks_mt.parquet
```

> [!TIP]
> You can additionally specify the Tokenizer path to filter out overlong samples:
> ```bash
> python3 scripts/prepare_sft_instruction_data.py --data_path  TowerBlocks-MT/data/train.parquet --output_path sft_towerblocks_mt.parquet --tokenizer_path path/to/tokenizer --max_len 1024
> ```

## Training

We employ [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for supervised fine-tuning.

### Setup
Install the LLaMA-Factory runtime environment according to the [documentation](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#getting-started).

### Dataset
- Place the SFT data in the `LLaMA-Factory/data` directory.
- Add the dataset information to `LLaMA-Factory/data/dataset_info.json`:
```json
{
    "sft_towerblocks_mt": {
        "file_name": "sft_towerblocks_mt.parquet",
        "columns": {
            "prompt": "prompt",
            "response": "response"
        }
    }
}
``` 

### Training Config

Create a yaml file to configure training hyperparameters, e.g., `qwen7b_full_sft_ds3.yaml`:
```yaml
### model
model_name_or_path: Qwen2.5-7B

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json

### dataset
dataset: sft_towerblocks_mt
template: chatml
cutoff_len: 2048
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/qwen-7b/full/eax_sft
logging_steps: 1
save_steps: 500000
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 8
gradient_accumulation_steps: 1
learning_rate: 7.0e-6
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
# warmup_steps: 200
bf16: true
# fp16: true
ddp_timeout: 180000000
```

> [!NOTE]  
> Depending on the number of computing devices, the total batch_size is calculated as `your_device_num * per_device_train_batch_size * gradient_accumulation_steps`.

> [!TIP]
> If your model does not support the special tokens defined in the chatml template, you need to add `add_special_tokens: <|im_start|>,<|im_end|>` to the yaml file.

### Run Training

Run the following command to start training:
```bash
FORCE_TORCHRUN=1 llamafactory-cli train path/to/qwen7b_full_sft_ds3.yaml 
```

> [!TIP]
> For Multiple Nodes training, refer to [Supervised Fine-Tuning on Multiple Nodes](https://github.com/hiyouga/LLaMA-Factory/tree/main/examples#supervised-fine-tuning-on-multiple-nodes).

## Next Steps

For the subsequent steps in the workflow, refer to the following documentation:
- [Reward Modeling](rm.md): build translation evaluation capabilities for the translation model.
- [x2x Optimization](xpo.md): curate x2x preference pair data with the translation model and reward model. Afterward, train the translation model with the curated data.