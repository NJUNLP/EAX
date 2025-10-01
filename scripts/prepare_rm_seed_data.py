import fire
import pandas as pd
from pathlib import Path
from utils.configs import valid_langs

def main(
    data_path: str = "TowerBlocks-MT/data/train.parquet",
    output_path: str = "TowerBlocks-MT/rm_seed_data",
):
    # Load the Parquet file
    df = pd.read_parquet(data_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    non_en_langs = valid_langs.copy()
    non_en_langs.remove("en")


    for lang in non_en_langs:
        df_lang2en = df[(df["src_lang"] == lang) & (df["trg_lang"] == "en")]
        df_en2lang = df[(df["src_lang"] == "en") & (df["trg_lang"] == lang)]

        # there are some duplicates src text in the dataset
        df_lang2en = df_lang2en.drop_duplicates(subset=["src_text"]).reset_index(drop=True)
        df_en2lang = df_en2lang.drop_duplicates(subset=["src_text"]).reset_index(drop=True)

        df_lang2en.to_parquet(output_path / f"{lang}2en.parquet")
        df_en2lang.to_parquet(output_path / f"en2{lang}.parquet")


if __name__ == "__main__":
    fire.Fire(main)