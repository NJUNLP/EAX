from sacrebleu.metrics import BLEU
from typing import Dict
import fire


def func_call(mt_list, ref_list, trg_lang: str = "", lowercase: bool = False):
    assert len(mt_list) == len(ref_list)
    bleu = BLEU(lowercase=lowercase, trg_lang=trg_lang, effective_order=True)
    corpus_score = bleu.corpus_score(mt_list, [ref_list])

    print("BLEU: ", corpus_score)

    sentence_scores = []
    for sys_sentence, ref_sentence in zip(mt_list, ref_list):
        sentence_scores.append(bleu.sentence_score(sys_sentence, [ref_sentence]).score)
    return {"system_score":corpus_score.score, "scores":sentence_scores} 

def main(mt_path: str, ref_path: str, trg_lang: str = "") -> Dict:

    with open(mt_path, "r") as f_mt, open(ref_path, "r") as f_ref:
        mt_list = [line.strip() for line in f_mt]
        ref_list = [line.strip() for line in f_ref]
    return func_call(mt_list, ref_list, trg_lang)



if __name__ == "__main__":
    fire.Fire(main)