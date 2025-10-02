from tqdm import tqdm
import pandas as pd
from vllm import LLM, SamplingParams
import glob
import os
import inference.run_vllm_eaxt_generation as run_vllm_eaxt_generation
from pathlib import Path
import fire

def main(
    model_path: str,
    data_path: str,
    output_path: str,
    max_new_tokens: int = 256,
    sampling_n: int = 1,
    chat_template: bool = True,
    part_id: int = None,
    device_num: int = 1,  # to specify the max part_id
    force_overwrite: bool = False,
    src_text_key: str = "src_text",
    ref_text_key: str = "ref_text",
    src_lang_key: str = "src_lang",
    trg_lang_key: str = "trg_lang",
    ref_lang_key: str = "ref_lang",
    out_key: str = "eaxt_text",
    temperature: float = 0.9,
    top_p: float = 0.6,
):

    output_path: Path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model = LLM(model=model_path)
    tokenizer = model.get_tokenizer()
    sampling_params = SamplingParams(
        n=1,  # may encounter bug for VLLM, use loop instead
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )

    file_list = glob.glob(os.path.join(data_path, "??2??.parquet"))
    file_list = sorted(file_list)

    if part_id is not None:
        assert part_id in list(range(device_num))
        file_list = file_list[part_id::device_num]

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print("found {} files.".format(len(file_list)))
    print("file_list:", [os.path.basename(file) for file in file_list])

    for file in tqdm(file_list):
        print("processing:", file)
        df = pd.read_parquet(file)
        original_order = df.index
        df["text_length"] = df[src_text_key].str.len()
        df = df.sort_values(by="text_length", ascending=True)
        df = df.drop(columns=["text_length"])

        if out_key not in df.columns or force_overwrite:

            src_list = df[src_text_key].tolist()
            ref_list = df[ref_text_key].tolist()
                
            src_langs = df[src_lang_key].tolist()
            trg_langs = df[trg_lang_key].tolist()
            ref_langs = df[ref_lang_key].tolist()

            output_texts_n = []
            for _ in range(sampling_n):
                output_texts = run_vllm_eaxt_generation.func_call(
                    model,
                    tokenizer,
                    src_list,
                    ref_list,
                    src_langs,
                    trg_langs,
                    ref_langs,
                    chat_template=chat_template,
                    max_new_tokens=max_new_tokens,
                    sampling_params=sampling_params,
                )
                output_texts = output_texts["response"]
                assert len(df) == len(output_texts)
                output_texts_n.append(output_texts)

            zipped_output_texts = list(zip(*output_texts_n))
            df[out_key] = zipped_output_texts

        df = df.reindex(original_order)

        file_name = os.path.basename(file)
        # Save the new DataFrame to the output path
        df.to_parquet(output_path / file_name)


if __name__ == "__main__":
    fire.Fire(main)