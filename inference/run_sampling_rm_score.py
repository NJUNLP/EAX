import eval.rm_score as rm_score

import torch
import pandas as pd
import glob
import os
from pathlib import Path
import fire

import numpy as np
from utils.helpers import flat_list, unflat_list, repeat_text


def run_rm(
    tokenizer,
    model,
    df,
    src_lang_key: str,
    trg_lang_key: str,
    src_text_key: str,
    mt_text_key: str,
    out_key: str,
    chat_template: bool = True,
    batch_size: int = 16,
):

    original_order = df.index

    df["text_length"] = df[src_text_key].str.len() + df[mt_text_key].str.len()
    df = df.sort_values(by="text_length", ascending=True)
    df = df.drop(columns=["text_length"])

    src_text = df[src_text_key].tolist()
    kd_texts_list = df[mt_text_key].tolist()
    kd_flattened_list, kd_text_count = flat_list(kd_texts_list)

    src_text = repeat_text(src_text, kd_text_count)

    src_lang = df.iloc[0][src_lang_key]
    kd_lang = df.iloc[0][trg_lang_key]

    # there is a risk of CUDA out of memory. So we try to run it again if it fails with a smaller batch size.
    while batch_size != 0:
        try:
            rm_out = rm_score.func_call(
                tokenizer=tokenizer,
                model=model,
                src_texts=src_text,
                mt_texts=kd_flattened_list,
                src_langs=src_lang,
                trg_langs=kd_lang,
                batch_size=batch_size,
                chat_template=chat_template,
            )
            break
        except torch.OutOfMemoryError:
            print(f"batch size {batch_size} failed. Trying with half batch size...")
            batch_size = batch_size // 2

    assert len(kd_flattened_list) == len(rm_out["scores"])
    rm_scores = unflat_list(rm_out["scores"], kd_text_count)
    df[out_key] = rm_scores
    df = df.reindex(original_order)
    return df


def main(
    data_path: str,
    rm_model_path: str,
    src_lang_key: str,
    trg_lang_key: str,
    src_text_key: str,
    mt_text_key: str,
    out_key: str,
    chat_template: bool = True,
    output_path: str = None,
    inplace: bool = False,
    part_id: int = None,
    device_num: int = 2,  # to specify the max part_id
    batch_size: int = 16,
):
    file_list = glob.glob(f"{data_path}/??2??.parquet")
    file_list = sorted(file_list)

    if part_id is not None:
        assert part_id in list(range(device_num))
        file_list = file_list[part_id::device_num]

    if output_path is None:
        assert inplace, "output_path must be specified when inplace is False"
        output_path = data_path

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    print("run list:", [os.path.basename(file) for file in file_list])

    print("loading model and tokenizer...")
    global tokenizer, model

    model, tokenizer = rm_score.load_model_and_tokenizer(rm_model_path)

    for file in file_list:
        print("running: ", os.path.basename(file))
        df = pd.read_parquet(file)
        df = run_rm(
            tokenizer,
            model,
            df,
            src_lang_key,
            trg_lang_key,
            src_text_key,
            mt_text_key,
            out_key,
            chat_template=chat_template,
            batch_size=batch_size,
        )
        df.to_parquet(output_path / os.path.basename(file))


if __name__ == "__main__":
    fire.Fire(main)