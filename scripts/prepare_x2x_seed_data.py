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
        nonen_texts = df_lang2en["src_text"].tolist() + df_en2lang["trg_text"].tolist()
        en_texts = df_lang2en["trg_text"].tolist() + df_en2lang["src_text"].tolist()

        for trg_lang in non_en_langs:
            if lang == trg_lang:
                continue
            df_x2x = pd.DataFrame({
                "src_text": nonen_texts,
                "ref_text": en_texts,
                "src_lang": lang,
                "trg_lang": trg_lang,
                "ref_lang": "en",
            })
            df_x2x = df_x2x.drop_duplicates(subset=["src_text"]).reset_index(drop=True)
            df_x2x.to_parquet(output_path / f"{lang}2{trg_lang}.parquet")


if __name__ == "__main__":
    fire.Fire(main)