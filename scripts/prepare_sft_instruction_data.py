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
import random
import fire
from functools import lru_cache
from utils.configs import lang_tag_map
from transformers import AutoTokenizer


# Set a fixed seed for reproducibility of random prompt selection and shuffling
random.seed(114514)

@lru_cache(maxsize=100)
def get_prompts(src_tag: str, trg_tag: str) -> tuple:
    """
    Returns a tuple of instruction prompt templates for translation 
    from source language tag (src_tag) to target language tag (trg_tag).
    The prompts use '{{}}' as a placeholder for the source text.
    """
    # Retrieve full language names from the map
    src_lang = lang_tag_map.get(src_tag, src_tag)
    trg_lang = lang_tag_map.get(trg_tag, trg_tag)
    
    # List of different instruction prompt templates
    prompts_list = (
        f"Translate the below text from {src_lang} to {trg_lang}.\nSource: {{}}\nTarget:",
        f"Source: {{}}\nYour task is to translate the following text from {src_lang} into {trg_lang}.\nTarget:",
        f"Please provide a translation from {src_lang} to {trg_lang} for the following text:\n{{}}\nTarget:",
        f"Source: {{}}\nTranslate the source text from {src_lang} to {trg_lang}.\nTarget:",
        f"Please translate the following text:\n{src_lang} Source: {{}}\n{trg_lang} Target:",
        f"Source: {{}}\nTranslate from {src_lang} to {trg_lang}.\nTarget:",
        f"Please translate this text from {src_lang} into {trg_lang}.\nSource: {{}}\nTarget:",
        f"Translate this {src_lang} text into {trg_lang}:\nSource: {{}}\nTranslation:",
        f"{src_lang} Source: {{}}\n{trg_lang} Translation:",
        f"Translate the following text from {src_lang} to {trg_lang}:\nText: {{}}\nAnswer:",
        f"Translate the following {src_lang} source text to {trg_lang}:\n{src_lang}: {{}}\n{trg_lang}:",
        f"Source: {{}}\nCan you translate the given text from {src_lang} into {trg_lang}?\nTarget:",
        f"Translate the text below from {src_lang} to {trg_lang}:\n{{}}\nTarget Translation:",
        f"Write a translation of the given text from {src_lang} to {trg_lang}.\n{src_lang}: {{}}\n{trg_lang}:",
        f"Source: {{}}\nGiven the text in {src_lang}, translate it into {trg_lang}.\nTarget:",
        f"Write the text in {src_lang} in {trg_lang}.\nSource: {{}}\nTarget:",
        f"From {src_lang} to {trg_lang}, translate the text:\nSource: {{}}\nTarget:",
        f"Source: {{}}\nProvide a translation of the given text from {src_lang} to {trg_lang}.\nTarget:",
        f"Make a translation of the given text from {src_lang} to {trg_lang}.\n{src_lang}: {{}}\n{trg_lang}:",
        f"{src_lang}: {{}}\n{trg_lang}:",
    )

    return prompts_list


def filter_by_token_length(df: pd.DataFrame, max_len: int, tokenizer) -> pd.DataFrame:
    """
    Filters the DataFrame rows based on the total number of tokens 
    (prompt + response) being less than or equal to max_len.
    Uses the provided Hugging Face tokenizer for token counting.
    """
    print(f"\n-> Calculating total token length (prompt + response) using tokenizer: {tokenizer.name_or_path}...")
    
    # Concatenate prompt and response for token counting
    # The tokenizer's __call__ method can accept a list of strings
    # and return the tokenized output, from which we can get the length of input_ids.
    texts = (df["prompt"] + df["response"]).tolist()
    
    # Tokenize in batches for efficiency (adjust batch_size if needed)
    tokenized_outputs = tokenizer(
        texts, 
        add_special_tokens=True, # Include special tokens in the length
        truncation=False, 
        padding=False # No padding needed for length counting
    )

    # Calculate token length for each sample
    df["total_length_tokens"] = [len(ids) for ids in tokenized_outputs["input_ids"]]
    
    initial_count = len(df)
    
    # Filter out rows where total token length exceeds max_len
    df_filtered = df.loc[df["total_length_tokens"] <= max_len]
    
    # Print filtering statistics
    filtered_count = len(df_filtered)
    removed_count = initial_count - filtered_count
    
    print(f"-> Filtering completed (max_len: {max_len} tokens).")
    print(f"   Samples before filtering: {initial_count}")
    print(f"   Samples removed: {removed_count}")
    
    # Drop the temporary column and reset index
    df_filtered = df_filtered.drop(columns=["total_length_tokens"])
    df_filtered = df_filtered.reset_index(drop=True)
    
    return df_filtered


def main(
    data_path: str,
    output_path: str,
    tokenizer_path: str, # New argument for HF tokenizer
    max_len: int = 1024, # Max sequence length in tokens
    src_key: str = "src",
    trg_key: str = "trg"
):
    """
    Main function to prepare SFT instruction data.

    Args:
        data_path (str): Path to the input Parquet file containing parallel data.
        output_path (str): Path for the output Parquet file.
        tokenizer_path (str): Hugging Face model path or local directory for the tokenizer.
        max_len (int): Maximum combined token length (prompt + response) for filtering.
        src_key (str): Key/prefix for source language columns (e.g., 'src').
        trg_key (str): Key/prefix for target language columns (e.g., 'trg').
    """
    print("--- SFT Instruction Data Preparation Script ---")
    
    # --- 1. Setup and Input Loading ---
    dir_name = os.path.dirname(output_path)
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    print(f"Output directory created/verified: {dir_name}")
    
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
    check_cols = [f"{src_key}_lang", f"{trg_key}_lang", f"{src_key}_text", f"{trg_key}_text"]
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
        src_lang = row[f"{src_key}_lang"]
        trg_lang = row[f"{trg_key}_lang"]
        
        # Get all prompt templates for the language pair
        prompts = get_prompts(src_lang, trg_lang)
        
        # Randomly select one prompt template
        prompt_template = random.choice(prompts)
        
        # Format the prompt with the source text
        prompt_list.append(prompt_template.format(row[f"{src_key}_text"]))
        
        # Collect the target text as the response
        response_list.append(row[f"{trg_key}_text"])

    # Create the new DataFrame with generated data
    df = pd.DataFrame({"prompt": prompt_list, "response": response_list})
    print(f"-> Instruction data generated for {len(df)} samples.")
    
    # --- 3. Token Length Filtering ---
    df = filter_by_token_length(df, max_len, tokenizer)
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