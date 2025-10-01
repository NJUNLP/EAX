
import eval.bleurt_eval as bleurt_eval
import pandas as pd
import glob
import os
from pathlib import Path
import numpy as np
import fire
from tqdm import tqdm



def flat_list(items_list: list) -> tuple[list, list]:
    item_count_list = []
    flattened_items_list = []
    for items in items_list:
        assert isinstance(items, list) or isinstance(items, np.ndarray)
        item_count_list.append(len(items))
        flattened_items_list.extend(items)

    return flattened_items_list, item_count_list

def unflat_list(flattened_items_list: list, item_count_list: list) -> list:
    assert sum(item_count_list) == len(flattened_items_list)
    unflattened_items_list = []
    start_idx = 0
    for item_count in item_count_list:
        end_idx = start_idx + item_count
        unflattened_items_list.append(flattened_items_list[start_idx:end_idx])
        start_idx = end_idx
    
    return unflattened_items_list

def repeat_text(text_list:list, repeat_count: list):
    assert len(text_list) == len(repeat_count)
    repeated_text_list = []
    for text, count in zip(text_list, repeat_count):
        repeated_text_list.extend([text] * count)
    return repeated_text_list



def run_scoring(df, bleurt_scorer, ref_key: str, mt_key: str, out_key: str):
    mt_texts = df[mt_key].tolist()
    ref_text = df[ref_key].tolist()
    mt_flattened_list, mt_text_count = flat_list(mt_texts)

    ref_text_list = repeat_text(ref_text, mt_text_count)
    assert len(mt_flattened_list) == len(ref_text_list)
    eval_out = bleurt_eval.func_call(
        None,
        mt_flattened_list,
        ref_text_list,
        scorer=bleurt_scorer,
    )
    assert len(mt_flattened_list) == len(eval_out["scores"])

    eval_out = unflat_list(eval_out["scores"], mt_text_count)
    df[out_key] = eval_out
    return df

def main(
    data_path: str,
    bleurt_path: str,
    ref_key: str,
    mt_key: str,
    out_key: str,
    output_path: str = None,
    inplace: bool = False,
    part_id: int = None,
    device_num: int = 1, # to specify the max part_id
):
    file_list = glob.glob(os.path.join(data_path, "??2??.parquet"))
    file_list = sorted(file_list)

    if part_id is not None:
        assert part_id in list(range(device_num))
        file_list = file_list[part_id::device_num]

    if output_path is None:
        assert inplace, "output_path must be specified when inplace is False"
        output_path = data_path

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    print("file list:", [os.path.basename(file) for file in file_list])
    
    print("loading bleurt scorer...")
    bleurt_scorer = bleurt_eval.score.BleurtScorer(bleurt_path)

    for file in tqdm(file_list):
        print("processing:", file)
        df = pd.read_parquet(file)
        df = run_scoring(df, bleurt_scorer, ref_key, mt_key, out_key)
        df.to_parquet(output_path / os.path.basename(file))


if __name__ == "__main__":
    fire.Fire(main)

