# x2x Optimization Recipe

Here we combine our translation model with the reward model to curate x2x preference pair data. Specifically, we run **English-Anchored x2x Translation (EAxT)** to generate high-quality translation candidates and **English-Anchored x2x Evaluation (EAxE)** to clean and construct the preference pairs.


## Data Preparation

Run the following command to download TowerBlocks-MT dataset if it is not locally available:
```bash
hf download double7/TowerBlocks-MT --repo-type dataset --local-dir TowerBlocks-MT
```

Split the dataset by language pairs; this allows us to leverage multiple GPUs for parallel processing:
```bash
python3 scripts/prepare_x2x_seed_data.py --data_path TowerBlocks-MT/data/train.parquet --output_path TowerBlocks-MT/x2x_seed_data
```

## Sampling Data by EAxT


Sample 4 translation candidates for each sample via EAxT, using 8 GPUs for parallel processing:
```bash
MODEL_PATH=path/to/eax_sft
DATA_PATH=TowerBlocks-MT/x2x_seed_data
OUT_KEY=eaxt_sampling_text
SAMPLING_N=4
DEVICE_NUM=8
for PART_ID in $(seq 0 $((DEVICE_NUM - 1))); do
    echo "Processing part $PART_ID"
    CUDA_VISIBLE_DEVICES=$PART_ID python3 inference/run_eaxt_sampling.py \
        --data_path $DATA_PATH \
        --output_path $DATA_PATH \
        --src_text_key src_text \
        --ref_text_key ref_text \
        --src_lang_key src_lang \
        --trg_lang_key trg_lang \
        --ref_lang_key ref_lang \
        --temperature 1.0 \
        --top_p 1.0 \
        --out_key $OUT_KEY \
        --model_path $MODEL_PATH  \
        --max_new_tokens 768 \
        --sampling_n $SAMPLING_N \
        --part_id $PART_ID \
        --device_num $DEVICE_NUM > eaxt_sampling.$PART_ID.log 2>&1 &
done
wait
```

> [!TIP]
> You can also use `inference/run_direct_mt_sampling.py` to generate x2x translation candidates directly.

Remove duplicate generated translation candidates:
```bash
python3 scripts/post_clean_sampling_data.py --data_path $DATA_PATH --text_key $OUT_KEY
```

## Scoring Data with EAxE

Use the reward model to calculate scores for all translation candidates, with parallel processing on 8 GPUs. Note that we use the reference text (the English reference) as the source text and the translation candidates as the target text.
```bash
DATA_PATH=TowerBlocks-MT/x2x_seed_data
RM_PATH=path/to/eax_rm
MT_KEY=eaxt_sampling_text
OUT_KEY=eaxt_sampling_rm_score
DEVICE_NUM=8
for PART_ID in $(seq 0 $((DEVICE_NUM - 1))); do
    CUDA_VISIBLE_DEVICES=$PART_ID python3 inference/run_sampling_rm_score.py \
        --data_path $DATA_PATH \
        --rm_model_path $RM_PATH \
        --src_lang_key ref_lang \
        --trg_lang_key trg_lang \
        --src_text_key ref_text \
        --mt_text_key $MT_KEY \
        --out_key $OUT_KEY \
        --inplace True \
        --part_id $PART_ID \
        --device_num $DEVICE_NUM > rm_score.$PART_ID.log 2>&1 &
done
wait
```

## Construct Preference Data

Construct preference data pairs based on reward model scores. The most critical parameter, `min_margin`, is used to control the minimum reward model score gap between preference data pairs.

```bash
python3 scripts/construct_preference_data.py \
    --data_path TowerBlocks-MT/x2x_seed_data \
    --output_path TowerBlocks-MT/x2x_pair_data.parquet \
    --mt_texts_key eaxt_sampling_text \
    --score_key eaxt_sampling_rm_score \
    --min_margin 10 \
    --src_text_key src_text \
    --src_lang_key src_lang \
    --trg_lang_key trg_lang \
    --ref_text_key ref_text \
    --ref_lang_key ref_lang
```

The script will print the distribution of the constructed preference data pairs across each language pair:
```
Language Distribution Matrix (src_lang → trg_lang)
============================================================
trg_lang     de     es     fr     it     ko     nl     pt     ru     zh   Total
src_lang                                                                       
de            0   5781   6385  10699  13839  12561   7049   9148   3760   69222
es         1749      0    687   1006   2266   2072    476   1442    555   10253
fr         2846    904      0   1873   3514   3436   1104   2361    976   17014
it         1581    405    566      0   1952   1915    516   1283    486    8704
ko         1027    439    556    912      0   1142    526    849    270    5721
nl         1052    596    701   1162   1456      0    696   1129    390    7182
pt         3267    621   1171   1936   4135   4084      0   2789    965   18968
ru         8713   4131   4447   7649  10399  10232   4430      0   2895   52896
zh         5911   3267   3446   5498   5791   6667   3532   5115      0   39227
Total     26146  16144  17959  30735  43352  42109  18329  24116  10297  229187

Percentage Distribution:
============================================================
trg_lang     de    es    fr     it     ko     nl    pt     ru    zh   Total
src_lang                                                                   
de         0.00  2.52  2.79   4.67   6.04   5.48  3.08   3.99  1.64   30.20
es         0.76  0.00  0.30   0.44   0.99   0.90  0.21   0.63  0.24    4.47
fr         1.24  0.39  0.00   0.82   1.53   1.50  0.48   1.03  0.43    7.42
it         0.69  0.18  0.25   0.00   0.85   0.84  0.23   0.56  0.21    3.80
ko         0.45  0.19  0.24   0.40   0.00   0.50  0.23   0.37  0.12    2.50
nl         0.46  0.26  0.31   0.51   0.64   0.00  0.30   0.49  0.17    3.13
pt         1.43  0.27  0.51   0.84   1.80   1.78  0.00   1.22  0.42    8.28
ru         3.80  1.80  1.94   3.34   4.54   4.46  1.93   0.00  1.26   23.08
zh         2.58  1.43  1.50   2.40   2.53   2.91  1.54   2.23  0.00   17.12
Total     11.41  7.04  7.84  13.41  18.92  18.37  8.00  10.52  4.49  100.00

Total samples: 229187
```

## Training

We employ [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for reward modeling.

### Setup
Install the LLaMA-Factory runtime environment according to the [documentation](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#getting-started).

### Dataset

Run the command below to add translation instructions to the translation pairs for reward modeling:
```bash
python3 scripts/prepare_pair_instruction_data.py --data_path  TowerBlocks-MT/x2x_pair_data.parquet --output_path x2x_towerblocks_mt.parquet --diverse_prompt
```

> [!TIP]
> You can additionally specify the Tokenizer path to filter out overlong samples:
> ```bash
> python3 scripts/prepare_pair_instruction_data.py --data_path  TowerBlocks-MT/rm_pair_data.parquet --output_path rm_towerblocks_mt.parquet --tokenizer_path path/to/tokenizer --max_len 1024 --diverse_prompt
> ```


- Place the preference pair data(`x2x_towerblocks_mt.parquet`) in the `LLaMA-Factory/data` directory.
- Add the dataset information to `LLaMA-Factory/data/dataset_info.json`:
```json
{
    "x2x_towerblocks_mt": {
        "file_name": "x2x_towerblocks_mt.parquet",
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

Create a yaml file to configure training hyperparameters, e.g., `qwen7b_full_dpo_ds3.yaml`:
```yaml
### model
model_name_or_path: path/to/eax_sft

### method
stage: dpo
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json
pref_beta: 0.4
pref_loss: sigmoid  # choices: [sigmoid (dpo), orpo, simpo]
pref_ftx: 2.0

### dataset
dataset: x2x_towerblocks_mt
template: chatml
cutoff_len: 2048
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/qwen-7b/full/eax_dpo
logging_steps: 1
save_steps: 1000
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 4
gradient_accumulation_steps: 1
learning_rate: 2.0e-7
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
# warmup_steps: 500
bf16: true
# fp16: true
ddp_timeout: 180000000
```

> [!NOTE]  
> Depending on the number of computing devices, the total batch_size is calculated as `your_device_num * per_device_train_batch_size * gradient_accumulation_steps`.

### Run Training

Run the following command to start DPO training:
```bash
FORCE_TORCHRUN=1 llamafactory-cli train path/to/qwen7b_full_dpo_ds3.yaml
```

> [!TIP]
> Lunch Multiple Nodes training:
> ```bash
> FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train path/to/qwen7b_full_dpo_ds3.yaml
> FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=1 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train path/to/qwen7b_full_dpo_ds3.yaml
> ```


## Next Steps

For the subsequent steps in the workflow, refer to the following documentation:
- [Evaluation](../README.md#evaluation-on-flores): evaluate models on FLORES and any other datasets.
