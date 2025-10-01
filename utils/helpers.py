import pandas as pd
from typing import Union

def filter_by_token_length(
    df: pd.DataFrame, 
    max_len: int, 
    tokenizer, 
    text_keys: Union[str, list[str]]
) -> pd.DataFrame:
    """
    Filters the DataFrame rows based on the total number of tokens 
    from specified columns being less than or equal to max_len.
    Uses the provided Hugging Face tokenizer for token counting.
    
    Args:
        df: Input DataFrame
        max_len: Maximum allowed token length
        tokenizer: Hugging Face tokenizer
        text_keys: Column name(s) to concatenate for token counting.
                  Can be a single string or list of strings.
    
    Returns:
        Filtered DataFrame with rows having token count <= max_len
    """
    # Normalize text_keys to always be a list
    if isinstance(text_keys, str):
        text_keys = [text_keys]
    
    # Validate that all specified columns exist in the DataFrame
    missing_cols = [col for col in text_keys if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Column(s) {missing_cols} not found in DataFrame. "
                        f"Available columns: {list(df.columns)}")
    
    print(f"\n-> Calculating total token length from columns {text_keys} "
          f"using tokenizer: {tokenizer.name_or_path}...")
    
    # Concatenate specified columns for token counting
    if len(text_keys) == 1:
        # Single column case
        texts = df[text_keys[0]].astype(str).tolist()
    else:
        # Multiple columns case - concatenate with space separator
        texts = df[text_keys].astype(str).agg(' '.join, axis=1).tolist()
    
    # Tokenize in batches for efficiency
    tokenized_outputs = tokenizer(
        texts, 
        add_special_tokens=True,  # Include special tokens in the length
        truncation=False, 
        padding=False  # No padding needed for length counting
    )

    # Calculate token length for each sample
    df = df.copy()  # Avoid modifying original DataFrame
    df["total_length_tokens"] = [len(ids) for ids in tokenized_outputs["input_ids"]]
    
    initial_count = len(df)
    
    # Filter out rows where total token length exceeds max_len
    df_filtered = df.loc[df["total_length_tokens"] <= max_len]
    
    # Print filtering statistics
    filtered_count = len(df_filtered)
    removed_count = initial_count - filtered_count
    
    print(f"-> Filtering completed (max_len: {max_len} tokens).")
    print(f"   Columns used: {text_keys}")
    print(f"   Samples before filtering: {initial_count}")
    print(f"   Samples removed: {removed_count}")
    print(f"   Samples remaining: {filtered_count}")
    
    # Drop the temporary column and reset index
    df_filtered = df_filtered.drop(columns=["total_length_tokens"])
    df_filtered = df_filtered.reset_index(drop=True)
    
    return df_filtered