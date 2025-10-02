from vllm import LLM, SamplingParams
from utils.prompts import get_eaxt_prompt
from inference.run_vllm_mt_generation import generate_batch

def func_call(
    model,
    tokenizer,
    src_list,
    ref_list,
    src_langs,
    trg_langs,
    ref_langs,
    chat_template=True,
    max_new_tokens=768,
    sampling_params=None,
):
    if isinstance(src_langs, str):
        src_langs = [src_langs] * len(src_list)
    if isinstance(trg_langs, str):
        trg_langs = [trg_langs] * len(src_list)
    if isinstance(ref_langs, str):
        ref_langs = [ref_langs] * len(src_list)

    if sampling_params is None:
        sampling_params = SamplingParams(
            n=1,
            temperature=0,
            max_tokens=max_new_tokens,
        )

    test_data = []

    for src_text, ref_text, src_lang, trg_lang, ref_lang in zip(src_list, ref_list, src_langs, trg_langs, ref_langs):
        test_data.append(
            {
                "src_text": src_text,
                "ref_text": ref_text,
                "src_lang": src_lang,
                "trg_lang": trg_lang,
                "ref_lang": ref_lang,
            }
        )

    example_prompt = get_eaxt_prompt(src_langs[0], trg_langs[0], ref_langs[0]).format(src_list[0], ref_list[0])

    print("Use prompt: {}".format(example_prompt))

    out_dict = {}

    input_texts = []
    for sample in test_data:
        src_text = sample["src_text"]
        ref_text = sample["ref_text"]
        
        src_lang = sample["src_lang"]
        trg_lang = sample["trg_lang"]
        ref_lang = sample["ref_lang"]

        input_text = get_eaxt_prompt(src_lang, trg_lang, ref_lang).format(src_text, ref_text)

        if chat_template:
            messages = [
                {"role": "user", "content": input_text},
            ]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        input_texts.append(input_text)

    responses = generate_batch(model, input_texts, sampling_params)

    out_dict["response"] = responses

    return out_dict
