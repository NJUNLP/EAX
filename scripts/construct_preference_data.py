import fire
from dataclasses import dataclass
from typing import Optional, Union
from functools import partial
import pandas as pd
import glob
import os
from pathlib import Path
from tqdm import tqdm


def load_df_list(data_path):
    file_list = glob.glob(os.path.join(data_path, "??2??.parquet"))
    file_list = sorted(file_list)
    df_list = [pd.read_parquet(file) for file in file_list]
    return df_list


def to_percentile(df, score_keys: Union[str, list[str]] = None):
    if isinstance(score_keys, str):
        score_keys = [score_keys]
    for key in score_keys:
        assert key in df.columns
        df[key] = df[key] * 100


@dataclass
class MT_Entry:
    mt_text: str
    score: float
    mt_ref_score: float = None
    mt_src_score: float = None
    weighted_score: float = None
    key_name: str = None


def construct_entry(
    mt_texts,
    scores,
    mt_ref_score=None,
    mt_src_score=None,
    key_name=None,
):
    if mt_ref_score is None:
        mt_ref_score = [None] * len(mt_texts)
    if mt_src_score is None:
        mt_src_score = [None] * len(mt_texts)

    return [
        MT_Entry(
            mt_text=mt_text,
            score=score,
            mt_ref_score=mt_ref_score,
            mt_src_score=mt_src_score,
            key_name=key_name,
        )
        for mt_text, score, mt_ref_score, mt_src_score in zip(
            mt_texts,
            scores,
            mt_ref_score,
            mt_src_score,
        )
    ]


def score_range_filter(
    data_entry: MT_Entry, max_score: float, min_score: float = None
) -> bool:
    if min_score is None:
        min_score = -1000
    if data_entry.mt_ref_score and data_entry.mt_ref_score < min_score:
        return False
    if data_entry.mt_ref_score and data_entry.mt_ref_score > max_score:
        return False
    return True


def min_score_filter(data_entry: MT_Entry, min_score: float) -> bool:
    if min_score is not None and data_entry.score < min_score:
        return False
    return True


def run_construct(
    df,
    mt_texts_key: str,
    score_key: str,
    maxnum: int = None,
    min_margin: int = 1,
    valid_score_range: tuple[float, float] = None,
    use_sampling_num: int = None,
    min_winner_score: Optional[float] = None,
    src_text_key: str = None,
    trg_text_key: str = None,
    src_lang_key: str = None,
    trg_lang_key: str = None,
    en_text_key: str = None,
):
    new_rows = []

    for i, row in df.iterrows():
        entry_list = construct_entry(
            row[mt_texts_key][:use_sampling_num],
            row[score_key][:use_sampling_num],
            key_name=mt_texts_key,
        )

        if valid_score_range is not None:
            # run pre filter
            entry_list = list(
                filter(
                    partial(
                        score_range_filter,
                        max_score=valid_score_range[1],
                        min_score=valid_score_range[0],
                    ),
                    entry_list,
                )
            )

        if len(entry_list) <= 1:
            continue

        # run compute weighted score
        for entry in entry_list:
            entry.weighted_score = entry.score

        winner_candidates: list[MT_Entry] = list(
            filter(
                partial(
                    min_score_filter,
                    min_score=min_winner_score,
                ),
                entry_list,
            )
        )

        if len(winner_candidates) == 0:
            continue

        loser_entry = min(entry_list, key=lambda x: x.weighted_score)

        winner_entry = None
        winner_margin = -10000
        for cand_entry in winner_candidates:
            cand_margin = cand_entry.weighted_score - loser_entry.weighted_score
            if cand_margin >= min_margin and cand_margin > winner_margin:
                winner_entry = cand_entry
                winner_margin = cand_margin

        if winner_entry is None:
            continue

        new_row = {
            "winner_text": winner_entry.mt_text,
            "winner_score": winner_entry.weighted_score,
            "winner_raw_score": winner_entry.score,
            "winner_key": winner_entry.key_name,
            "loser_text": loser_entry.mt_text,
            "loser_score": loser_entry.weighted_score,
            "loser_raw_score": loser_entry.score,
            "loser_key": loser_entry.key_name,
            "margin": winner_entry.weighted_score - loser_entry.weighted_score,
        }
        if src_text_key is not None:
            new_row["src_text"] = row[src_text_key]
        if trg_text_key is not None:
            new_row["trg_text"] = row[trg_text_key]
        if src_lang_key is not None:
            new_row["src_lang"] = row[src_lang_key]
        if trg_lang_key is not None:
            new_row["trg_lang"] = row[trg_lang_key]
        if en_text_key is not None:
            new_row["en_text"] = row[en_text_key]

        new_rows.append(new_row)

    if maxnum is not None:
        # Sort by margin in descending order
        sorted_rows = sorted(new_rows, key=lambda x: x["margin"], reverse=True)

        # Select up to maxnum samples
        selected_rows = sorted_rows[:maxnum]
    else:
        selected_rows = new_rows

    # Create the final DataFrame
    final_df = pd.DataFrame(selected_rows).drop(columns=["margin"])

    return final_df

def print_lang_dist(df):
    # Create a pivot table to get the matrix format
    lang_matrix = df.groupby(['src_lang', 'trg_lang']).size().unstack(fill_value=0)
    
    # Reorder columns to put 'en' first
    columns = lang_matrix.columns.tolist()
    if 'en' in columns:
        columns.remove('en')
        columns = ['en'] + sorted(columns)
    else:
        columns = sorted(columns)
    lang_matrix = lang_matrix.reindex(columns=columns)
    
    # Reorder rows to put 'en' first
    rows = lang_matrix.index.tolist()
    if 'en' in rows:
        rows.remove('en')
        rows = ['en'] + sorted(rows)
    else:
        rows = sorted(rows)
    lang_matrix = lang_matrix.reindex(index=rows)
    
    # Calculate row sums (src_lang totals)
    row_sums = lang_matrix.sum(axis=1)
    
    # Calculate column sums (trg_lang totals)
    col_sums = lang_matrix.sum(axis=0)
    
    # Add row sums as a new column
    lang_matrix['Total'] = row_sums
    
    # Add column sums as a new row
    lang_matrix.loc['Total'] = list(col_sums) + [df.shape[0]]
    
    print("Language Distribution Matrix (src_lang → trg_lang)")
    print("=" * 60)
    print(lang_matrix)
    
    print("\nPercentage Distribution:")
    print("=" * 60)
    percentage_matrix = (lang_matrix / df.shape[0] * 100).round(2)
    print(percentage_matrix)
    
    print(f"\nTotal samples: {df.shape[0]}")


def construct_rm_data(
    data_path: str,
    output_path: str,
    mt_texts_key: str,
    score_key: str,
    min_margin: int = 1,
    valid_score_range: Optional[tuple[float, float]] = None,
    limit_for_each_lang_pair: Optional[int] = None,
    use_sampling_num: Optional[int] = None,
    min_winner_score: Optional[float] = None,
    src_text_key: Optional[str] = None,
    trg_text_key: Optional[str] = None,
    src_lang_key: Optional[str] = None,
    trg_lang_key: Optional[str] = None,
    en_text_key: Optional[str] = None,
):
    df_list = load_df_list(data_path)

    for df in df_list:
        to_percentile(df, score_key)

    print("all sample:", sum([len(df) for df in df_list]))

    pair_data_df_list = [
        run_construct(
            df,
            mt_texts_key,
            score_key,
            maxnum=limit_for_each_lang_pair,
            min_margin=min_margin,
            valid_score_range=valid_score_range,
            use_sampling_num=use_sampling_num,
            min_winner_score=min_winner_score,
            src_text_key=src_text_key,
            trg_text_key=trg_text_key,
            src_lang_key=src_lang_key,
            trg_lang_key=trg_lang_key,
            en_text_key=en_text_key,
        )
        for df in tqdm(df_list, desc="Processing language pairs")
    ]

    df = pd.concat(pair_data_df_list, ignore_index=True)
    df = df.sample(frac=1, random_state=114514).reset_index(drop=True)
    print_lang_dist(df)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)


if __name__ == "__main__":
    fire.Fire(construct_rm_data)