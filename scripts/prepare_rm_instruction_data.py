"""
Script for preparing Reward Modeling instruction data 
from a Parquet file of preference pair data.

It generates instruction-response pairs by randomly selecting a prompt template,
filters the data based on the total token length (prompt + chosen response) 
using a specified Hugging Face tokenizer, and then shuffles and saves 
the final dataset to a new Parquet file.
"""

import pandas as pd
from pathlib import Path
import os
import fire
from transformers import AutoTokenizer
from utils.prompts import get_eval_prompt, get_diverse_prompt
from utils.helpers import filter_by_token_length



def main(
    data_path: str,
    output_path: str,
    tokenizer_path: str = None,
    max_len:int = 1024,
    src_text_key:str = "src_text",
    src_lang_key:str = "src_lang",
    trg_lang_key:str = "trg_lang",
    chosen_text_key:str = "winner_text",
    rejected_text_key:str = "loser_text",
    diverse_prompt: bool = False
):
    # --- 1. Setup and Input Loading ---
    dir_name = os.path.dirname(output_path)
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    print(f"Output directory created/verified: {dir_name}")
    
    tokenizer = None
    if tokenizer_path:
        # Load Hugging Face Tokenizer
        print(f"\n-> Loading tokenizer from: {tokenizer_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            print("-> Tokenizer loaded successfully.")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Exiting due to tokenizer loading failure.")
            return

    # Load data
    print(f"\n-> Loading data from: {data_path}")
    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        print(f"Error reading Parquet file at {data_path}: {e}")
        return

    # Check required columns
    check_cols = [src_lang_key, trg_lang_key, src_text_key, chosen_text_key, rejected_text_key]
    for col in check_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in input DataFrame columns: {list(df.columns)}")
            return
    
    initial_samples = len(df)
    print(f"-> Initial number of samples loaded: {initial_samples}")
    print(f"-> Filtering will be applied with max token length: {max_len}")
    
    # --- 2. Prompt Generation ---
    print("\n-> Generating instruction prompts...")
    prompt_list = []
    chosen_list = []
    rejected_list = []
    
    # Loop over the DataFrame to generate prompt-chosen-rejected pairs
    for _, row in df.iterrows():
        src_lang = row[src_lang_key]
        trg_lang = row[trg_lang_key]
        
        if diverse_prompt:
            prompt = get_diverse_prompt(src_lang, trg_lang)
        else:
            prompt = get_eval_prompt(src_lang, trg_lang)
        prompt_list.append(prompt.format(row[src_text_key]))
        chosen_list.append(row[chosen_text_key])
        rejected_list.append(row[rejected_text_key])
        

    df = pd.DataFrame({"prompt": prompt_list, "chosen": chosen_list, "rejected": rejected_list})
    print(f"-> Instruction data generated for {len(df)} samples.")
    
    # --- 3. Token Length Filtering ---
    if tokenizer:
        df = filter_by_token_length(df, max_len, tokenizer, ["prompt", "chosen"])
    
    final_samples = len(df)
    print(f"\n-> Final number of samples after filtering: {final_samples}")
    # --- 4. Shuffle and Save ---
    print("\n-> Shuffling and saving final dataset...")
    # Shuffle the DataFrame rows in place
    df = df.sample(frac=1, random_state=114514).reset_index(drop=True)
    
    # Save to Parquet
    try:
        df.to_parquet(output_path)
        print(f"-> Successfully saved final dataset to: {output_path}")
    except Exception as e:
        print(f"Error saving Parquet file to {output_path}: {e}")

    print("\n--- Script finished ---")


if __name__ == "__main__":
    fire.Fire(main)