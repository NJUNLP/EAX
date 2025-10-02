from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from trl import AutoModelForCausalLMWithValueHead

import os
import torch
from tqdm import tqdm
import pandas as pd

from typing import Union, Dict
from utils.prompts import get_eval_prompt


def load_valuehead_params(path_or_repo_id: str) -> Dict[str, torch.Tensor]:
    r"""
    Loads value head parameters from Hugging Face Hub or local disk.

    Returns: dict with keys `v_head.summary.weight` and `v_head.summary.bias`.
    """
    err_text = ""

    from safetensors import safe_open

    try:
        vhead_file = os.path.join(path_or_repo_id, "value_head.safetensors")
        with safe_open(vhead_file, framework="pt", device="cpu") as f:
            return {key: f.get_tensor(key) for key in f.keys()}
    except Exception as err:
        err_text = str(err)

    print(
        f"Provided path ({path_or_repo_id}) does not contain value head weights: {err_text}."
    )
    print(
        "Ignore the above message if you are not resuming the training of a value head model."
    )
    return None


def get_prompt_response_input(
    tokenizer, src_lang, trg_lang, src_text, mt_text, chat_template=True
):
    prompt = get_eval_prompt(src_lang, trg_lang).format(src_text)

    if chat_template:
        messages = [
            {"role": "user", "content": prompt},
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt += " "

    prompt_response = f"{prompt}{mt_text}{tokenizer.eos_token}"
    return prompt_response


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]



def load_model_and_tokenizer(model_path) -> tuple[AutoModelForCausalLMWithValueHead, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
    )

    vhead_params = load_valuehead_params(model_path)

    if vhead_params is not None:
        model.load_state_dict(vhead_params, strict=False)
        print(f"Loaded valuehead from checkpoint: {model_path}")
    else:
        raise RuntimeError("Value head not found in the model.")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = model.eval()
    return model, tokenizer


def func_call(
    src_texts: list,
    mt_texts: list,
    src_langs: Union[str, list],
    trg_langs: Union[str, list],
    batch_size: int = 16,
    chat_template: bool = True,
    tokenizer=None,
    model=None,
    model_path: str = None,
):

    if model_path is None:
        assert model is not None and tokenizer is not None
    else:
        tokenizer, model = load_model_and_tokenizer(model_path)

    if isinstance(src_langs, str):
        src_langs = [src_langs] * len(src_texts)
    if isinstance(trg_langs, str):
        trg_langs = [trg_langs] * len(src_texts)

    assert len(src_texts) == len(mt_texts)

    test_data = []

    for src_text, mt_text, src_lang, trg_lang in zip(
        src_texts, mt_texts, src_langs, trg_langs
    ):
        test_data.append(
            {
                "src_text": src_text,
                "mt_text": mt_text,
                "src_lang": src_lang,
                "trg_lang": trg_lang,
            }
        )

    example_prompt = get_prompt_response_input(
        tokenizer,
        test_data[0]["src_lang"],
        test_data[0]["trg_lang"],
        test_data[0]["src_text"],
        test_data[0]["mt_text"],
    )

    print("Use prompt: {}".format(example_prompt))

    out_list = []

    for batch_samples in tqdm(batch(test_data, batch_size), desc="Processing batches"):
        input_texts = []
        for sample in batch_samples:
            assert (
                "src_text" in sample
                and "mt_text" in sample
                and "src_lang" in sample
                and "trg_lang" in sample
            )
            src_text = sample["src_text"]
            mt_text = sample["mt_text"]
            src_lang = sample["src_lang"]
            trg_lang = sample["trg_lang"]

            input_text = get_prompt_response_input(
                tokenizer,
                src_lang,
                trg_lang,
                src_text,
                mt_text,
                chat_template=chat_template,
            )

            input_texts.append(input_text)

        inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to("cuda")

        with torch.no_grad():
            values = model(**inputs, return_dict=True, use_cache=False)[-1]

        scores = values.gather(
            dim=-1, index=(inputs["attention_mask"].sum(dim=-1, keepdim=True) - 1)
        )

        scores = scores.squeeze(-1).tolist()
        out_list.extend(scores)

    eval_out = {"scores": out_list}

    return eval_out
