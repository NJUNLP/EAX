from typing import Literal, Dict, Any, List
import json
import os
from utils.configs import valid_langs, flores_langcode_map
import fire
import eval.run_eval as run_eval


def prepare_infer_data(infer_data_path: str, test_data_path: str, split: str, valid_langs: List[str]
):
    flores_data = {}
    for lang in valid_langs:
        data_file = os.path.join(test_data_path, split, f"{flores_langcode_map[lang]}.{split}")
        with open(data_file, "r", encoding="utf-8") as f:
            flores_data[lang] = [line.strip() for line in f.readlines()]

    infer_data = []

    for src_lang in valid_langs:
        for trg_lang in valid_langs:
            if src_lang == trg_lang:
                continue
            for src_text, trg_text in zip(flores_data[src_lang], flores_data[trg_lang]):
                infer_data.append({
                    "src_lang": src_lang,
                    "trg_lang": trg_lang,
                    "src_text": src_text,
                    "trg_text": trg_text,
                })

    with open(infer_data_path, "w", encoding="utf-8") as f:
        json.dump(infer_data, f, indent=2, ensure_ascii=False)



def get_data_identifier(config: dict) -> str:
    return f"{config['model_name']}-dataset_{config['dataset']}-split_{config['split']}"


def run(
    model_path: str,
    model_name: str = "",
    test_data_path: str = "flores200_dataset",
    split: str = "devtest",
    metrics: list[str] = ["bleurt", "sacrebleu", "comet"],
    bleurt_path: str = "BLEURT-20",
    comet_path: str = "wmt22-comet-da/checkpoints/model.ckpt",
    log_to_wandb: bool = True,
):
    if model_name == "":
        model_name = os.path.basename(model_path)

    # Prepare wandb config dictionary
    config = {
        "model_name": model_name,
        "langs": valid_langs,
        "test_data_path": test_data_path,
        "metrics": metrics,
        "dataset": f"flores200-{split}",
        "split": split,
    }

    run_identifier = get_data_identifier(config)
    infer_data_path = f"infer_data.{run_identifier}.json"

    print(f"Preparing inference data at: {infer_data_path}")
    prepare_infer_data(infer_data_path, test_data_path, split, valid_langs)

    run_eval.run(
        model_path,
        infer_data_path,
        metrics,
        bleurt_path,
        comet_path,
        log_to_wandb,
        config,
    )



if __name__ == "__main__":
    fire.Fire(run)