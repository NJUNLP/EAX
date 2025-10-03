# RM Recipe


Based on the [SFT](sft.md) translation model, we build translation evaluation capabilities for it through Reward Modeling, which is then used in the subsequent English-Anchored x2x Evaluation.


## Data Preparation

Run the following command to download TowerBlocks-MT dataset if it is not locally available:
```bash
hf download double7/TowerBlocks-MT --repo-type dataset --local-dir TowerBlocks-MT
```

Split the dataset by language pairs; this allows us to leverage multiple GPUs for parallel processing:
```bash
python3 scripts/prepare_rm_seed_data.py --data_path TowerBlocks-MT/data/train.parquet --output_path TowerBlocks-MT/rm_seed_data
```

## Sampling Data

Sample 4 translation candidates for each sample, using 8 GPUs for parallel processing:
```bash
MODEL_PATH=path/to/eax_sft
DATA_PATH=TowerBlocks-MT/rm_seed_data
OUT_KEY=rm_sampling_text
SAMPLING_N=4
DEVICE_NUM=8
for PART_ID in $(seq 0 $((DEVICE_NUM - 1))); do
    echo "Processing part $PART_ID"
    CUDA_VISIBLE_DEVICES=$PART_ID python3 inference/run_direct_mt_sampling.py \
        --data_path $DATA_PATH \
        --output_path $DATA_PATH \
        --src_text_key src_text \
        --src_lang_key src_lang \
        --trg_lang_key trg_lang \
        --temperature 1.0 \
        --top_p 1.0 \
        --out_key $OUT_KEY \
        --model_path $MODEL_PATH  \
        --max_new_tokens 768 \
        --sampling_n $SAMPLING_N \
        --part_id $PART_ID \
        --device_num $DEVICE_NUM > rm_sampling.$PART_ID.log 2>&1 &
done
wait
```

Remove duplicate generated translation candidates:
```bash
python3 scripts/post_clean_sampling_data.py --data_path $DATA_PATH --text_key $OUT_KEY
```

## Scoring Data with BLEURT

### Setup

First, ensure the [BLEURT](https://github.com/google-research/bleurt) library and its model are locally available.

> [!TIP]
> You can also run `pip install -e '.[eval]'` to install the BLEURT library (model not included).

### Run Scoring
Use the BLEURT model to calculate BLEURT scores for all translation candidates, with parallel processing on 8 GPUs:
```bash
DATA_PATH=TowerBlocks-MT/rm_seed_data
BLEURT_PATH=BLEURT-20
MT_KEY=rm_sampling_text
OUT_KEY=rm_sampling_bleurt_score
DEVICE_NUM=8
for PART_ID in $(seq 0 $((DEVICE_NUM - 1))); do
    CUDA_VISIBLE_DEVICES=$PART_ID python3 inference/run_sampling_bleurt_score.py \
        --data_path $DATA_PATH \
        --bleurt_path $BLEURT_PATH \
        --ref_key trg_text \
        --mt_key $MT_KEY \
        --out_key $OUT_KEY \
        --inplace True \
        --part_id $PART_ID \
        --device_num $DEVICE_NUM > bleurt_score.$PART_ID.log 2>&1 &
done
wait
```

## Construct Preference Data

Construct preference data pairs based on BLEURT scores. The most critical parameter, `min_margin`, is used to control the minimum BLEURT score gap between preference data pairs. Additionally, we set `valid_score_range` to exclude outliers (we observed that BLEURT scores may be less than 0 or greater than 100).

```bash
python3 scripts/construct_preference_data.py \
    --data_path TowerBlocks-MT/rm_seed_data \
    --output_path TowerBlocks-MT/rm_pair_data.parquet \
    --mt_texts_key rm_sampling_text \
    --score_key rm_sampling_bleurt_score \
    --min_margin 20 \
    --valid_score_range [20,95] \
    --min_winner_score 70 \
    --src_text_key src_text \
    --trg_text_key trg_text \
    --src_lang_key src_lang \
    --trg_lang_key trg_lang \
    --to_percentile True # multiply the score by 100
```

The script will print the distribution of the constructed preference data pairs across each language pair:
```
Language Distribution Matrix (src_lang → trg_lang)
============================================================
trg_lang    en     de    es    fr    it    ko    nl    pt     ru    zh  Total
src_lang                                                                     
en           0  12351  1820  3607  2719  1190  1879  3853  11274  2409  41102
de        3607      0     0     0     0     0     0     0      0     0   3607
es          92      0     0     0     0     0     0     0      0     0     92
fr         551      0     0     0     0     0     0     0      0     0    551
it         105      0     0     0     0     0     0     0      0     0    105
ko         174      0     0     0     0     0     0     0      0     0    174
nl         125      0     0     0     0     0     0     0      0     0    125
pt          97      0     0     0     0     0     0     0      0     0     97
ru        2033      0     0     0     0     0     0     0      0     0   2033
zh        1362      0     0     0     0     0     0     0      0     0   1362
Total     8146  12351  1820  3607  2719  1190  1879  3853  11274  2409  49248

Percentage Distribution:
============================================================
trg_lang     en     de   es    fr    it    ko    nl    pt     ru    zh   Total
src_lang                                                                      
en         0.00  25.08  3.7  7.32  5.52  2.42  3.82  7.82  22.89  4.89   83.46
de         7.32   0.00  0.0  0.00  0.00  0.00  0.00  0.00   0.00  0.00    7.32
es         0.19   0.00  0.0  0.00  0.00  0.00  0.00  0.00   0.00  0.00    0.19
fr         1.12   0.00  0.0  0.00  0.00  0.00  0.00  0.00   0.00  0.00    1.12
it         0.21   0.00  0.0  0.00  0.00  0.00  0.00  0.00   0.00  0.00    0.21
ko         0.35   0.00  0.0  0.00  0.00  0.00  0.00  0.00   0.00  0.00    0.35
nl         0.25   0.00  0.0  0.00  0.00  0.00  0.00  0.00   0.00  0.00    0.25
pt         0.20   0.00  0.0  0.00  0.00  0.00  0.00  0.00   0.00  0.00    0.20
ru         4.13   0.00  0.0  0.00  0.00  0.00  0.00  0.00   0.00  0.00    4.13
zh         2.77   0.00  0.0  0.00  0.00  0.00  0.00  0.00   0.00  0.00    2.77
Total     16.54  25.08  3.7  7.32  5.52  2.42  3.82  7.82  22.89  4.89  100.00

Total samples: 49248
```

> [!TIP]
> If you need to balance the number of samples between x2en and en2x, you can use `--limit_for_each_lang_pair [int]` to restrict the number of preference pairs extracted from each language pair.


## Training

We employ [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for reward modeling.

### Setup
Install the LLaMA-Factory runtime environment according to the [documentation](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#getting-started).

### Dataset

Run the command below to add translation instructions to the translation pairs for reward modeling:
```bash
python3 scripts/prepare_pair_instruction_data.py --data_path  TowerBlocks-MT/rm_pair_data.parquet --output_path rm_towerblocks_mt.parquet
```

> [!TIP]
> You can additionally specify the Tokenizer path to filter out overlong samples:
> ```bash
> python3 scripts/prepare_pair_instruction_data.py --data_path  TowerBlocks-MT/rm_pair_data.parquet --output_path rm_towerblocks_mt.parquet --tokenizer_path path/to/tokenizer --max_len 1024
> ```



- Place the preference pair data(`rm_towerblocks_mt.parquet`) in the `LLaMA-Factory/data` directory.
- Add the dataset information to `LLaMA-Factory/data/dataset_info.json`:
```json
{
    "rm_towerblocks_mt": {
        "file_name": "rm_towerblocks_mt.parquet",
        "ranking": true,
        "columns": {
        "prompt": "prompt",
        "chosen": "chosen",
        "rejected": "rejected"
        }
    }
}
```

### Training Config

Create a yaml file to configure training hyperparameters, e.g., `qwen7b_full_rm_ds3.yaml`:
```yaml
### model
model_name_or_path: path/to/eax_sft

### method
stage: rm
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json

### dataset
dataset: rm_towerblocks_mt
template: chatml
cutoff_len: 2048
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/qwen-7b/full/eax_rm
logging_steps: 1
save_steps: 500000
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 4
gradient_accumulation_steps: 1
learning_rate: 4.0e-6
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.04
per_device_eval_batch_size: 4
eval_strategy: steps
eval_steps: 100
```

> [!NOTE]  
> Depending on the number of computing devices, the total batch_size is calculated as `your_device_num * per_device_train_batch_size * gradient_accumulation_steps`.

### Run Training

Run the following command to start training:
```bash
FORCE_TORCHRUN=1 llamafactory-cli train path/to/qwen7b_full_rm_ds3.yaml
```

> [!TIP]
> Lunch Multiple Nodes training:
> ```bash
> FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train path/to/qwen7b_full_rm_ds3.yaml
> FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=1 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train path/to/qwen7b_full_rm_ds3.yaml
> ```


## Next Steps

For the subsequent steps in the workflow, refer to the following documentation:
- [x2x Optimization](xpo.md): curate x2x preference pair data with the translation model and reward model. Afterward, train the translation model with the curated data.
- [Evaluation](../README.md#evaluation-on-flores): evaluate models on FLORES and any other datasets.
