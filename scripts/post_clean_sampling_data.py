import pandas as pd
import glob
import os
import fire


special_token = "</s>"

def remove_special_token_from_list(texts: list[str]):
    output = []
    for text in texts:
        if special_token in text:
            text = text[: text.index(special_token)]
        output.append(text)
    return output


def main(data_path: str, text_key: str):
    file_list = glob.glob(os.path.join(data_path, "??2??.parquet"))
    file_list = sorted(file_list)

    df_list = [pd.read_parquet(file) for file in file_list]

    for file, df in zip(file_list, df_list):
        item_count = df[text_key].apply(lambda x: len(x)).sum()
        
        df[text_key] = df[text_key].apply(lambda x: remove_special_token_from_list(x))
        df[text_key] = df[text_key].apply(lambda x: list(set(x)))
        new_item_count = df[text_key].apply(lambda x: len(x)).sum()

        print(
            "file: {}, removed item: {}/{}".format(
                os.path.basename(file),
                item_count - new_item_count,
                item_count,
            )
        )
        df.to_parquet(file)


if __name__ == "__main__":
    fire.Fire(main)