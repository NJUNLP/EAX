from utils.configs import lang_tag_map
from functools import lru_cache
import random

random.seed(114514)

@lru_cache(maxsize=100)
def _get_prompts(src_tag: str, trg_tag: str) -> tuple:
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



def get_eval_prompt(src_tag: str, trg_tag: str) -> str:
    src_lang = lang_tag_map.get(src_tag, src_tag)
    trg_lang = lang_tag_map.get(trg_tag, trg_tag)
    prompt = f"Translate the following text from {src_lang} into {trg_lang}:\n{src_lang}: {{}}\n{trg_lang}:"

    return prompt


def get_diverse_prompt(src_tag: str, trg_tag: str) -> str:
    """
    Returns a diverse prompt template for translation from source language tag (src_tag) 
    to target language tag (trg_tag). The prompt uses '{{}}' as a placeholder for the source text.
    """
    prompts_list = _get_prompts(src_tag, trg_tag)
    prompt = random.choice(prompts_list)
    return prompt


def get_eaxt_prompt(src_tag: str, trg_tag: str, ref_tag: str) -> str:
    src_lang = lang_tag_map.get(src_tag, src_tag)
    trg_lang = lang_tag_map.get(trg_tag, trg_tag)
    ref_lang = lang_tag_map.get(ref_tag, ref_tag)
    prompt = f"Given a {src_lang} text and its {ref_lang} version as a reference, translate the source text into {trg_lang}.\n{src_lang} Source: {{}}\n{ref_lang} Reference: {{}}\n{trg_lang} Translation:"
    
    return prompt
