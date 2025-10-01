# prepare_sft_instruction_data.py

"""
Script for preparing Supervised Fine-Tuning (SFT) instruction data 
from a Parquet file of parallel translation data.

It generates instruction-response pairs by randomly selecting a prompt template,
filters the data based on the total token length (prompt + response) 
using a specified Hugging Face tokenizer, and then shuffles and saves 
the final dataset to a new Parquet file.
"""

import pandas as pd
from pathlib import Path
import os
import fire
from transformers import AutoTokenizer
from utils.prompts import get_diverse_prompt
from utils.helpers import filter_by_token_length



def main(
    data_path: str,
    output_path: str,
    tokenizer_path: str = None,
    max_len: int = 1024,
    src_text_key: str = "src_text",
    trg_text_key: str = "trg_text",
    src_lang_key: str = "src_lang",
    trg_lang_key: str = "trg_lang",
):
    """
    Main function to prepare SFT instruction data.

    Args:
        data_path (str): Path to the input Parquet file containing parallel data.
        output_path (str): Path for the output Parquet file.
        tokenizer_path (str): Hugging Face model path or local directory for the tokenizer.
        max_len (int): Maximum combined token length (prompt + response) for filtering.
        src_text_key (str): Key/prefix for source language text (e.g., 'src_text').
        trg_text_key (str): Key/prefix for target language text (e.g., 'trg_text').
        src_lang_key (str): Key/prefix for source language columns (e.g., 'src_lang').
        trg_lang_key (str): Key/prefix for target language columns (e.g., 'trg_lang').
    """
    print("--- SFT Instruction Data Preparation Script ---")
    
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
    check_cols = [src_lang_key, trg_lang_key, src_text_key, trg_text_key]
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
    response_list = []
    
    # Loop over the DataFrame to generate prompt-response pairs
    for _, row in df.iterrows():
        src_lang = row[src_lang_key]
        trg_lang = row[trg_lang_key]
        
        # Randomly select one prompt template
        prompt_template = get_diverse_prompt(src_lang, trg_lang)
        
        # Format the prompt with the source text
        prompt_list.append(prompt_template.format(row[src_text_key]))
        
        # Collect the target text as the response
        response_list.append(row[trg_text_key])

    # Create the new DataFrame with generated data
    df = pd.DataFrame({"prompt": prompt_list, "response": response_list})
    print(f"-> Instruction data generated for {len(df)} samples.")
    
    # --- 3. Token Length Filtering ---
    if tokenizer:
        df = filter_by_token_length(df, max_len, tokenizer, ["prompt", "response"])
    
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
    # Example usage with required arguments:
    # python prepare_sft_instruction_data.py --data_path=input.parquet --output_path=output.parquet --tokenizer_path=meta-llama/Llama-2-7b-hf
    # For local tokenizer:
    # python prepare_sft_instruction_data.py --data_path=input.parquet --output_path=output.parquet --tokenizer_path=/path/to/local/tokenizer/dir
    fire.Fire(main)