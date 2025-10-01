from typing import Dict
from bleurt import score
import fire

def func_call(bleurt_path , mt_list, ref_list, scorer=None):
    assert len(mt_list) == len(ref_list)

    if scorer is None:
        assert bleurt_path is not None, "bleurt_path is None"
        scorer = score.BleurtScorer(bleurt_path)

    scores = scorer.score(references=ref_list, candidates=mt_list, batch_size=100)
    system_score = sum(scores) / len(scores)

    print ("bleurt: ", system_score)

    return {"scores": scores, "system_score": system_score}



def main(mt_path: str, ref_path: str, bleurt_path: str = "BLEURT-20") -> Dict:

    with open(mt_path, "r") as f_mt, open(ref_path, "r") as f_ref:
        mt_list = [line.strip() for line in f_mt]
        ref_list = [line.strip() for line in f_ref]
        
    return func_call(bleurt_path, mt_list, ref_list)



if __name__ == "__main__":
    fire.Fire(main)