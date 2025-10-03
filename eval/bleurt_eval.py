from typing import Dict
from bleurt import score
import fire
import subprocess
import os

def func_call(bleurt_path , mt_list, ref_list, scorer=None):
    assert len(mt_list) == len(ref_list)

    if scorer is None:
        assert bleurt_path is not None, "bleurt_path is None"
        scorer = score.BleurtScorer(bleurt_path)

    scores = scorer.score(references=ref_list, candidates=mt_list, batch_size=100)
    system_score = sum(scores) / len(scores)

    print ("bleurt: ", system_score)

    return {"scores": scores, "system_score": system_score}


def cli_call(bleurt_path , mt_list, ref_list):
    with open("_temp.mt", "w") as f_mt, open("_temp.ref", "w") as f_ref:
        for mt, ref in zip(mt_list, ref_list):
            f_mt.write(mt + "\n")
            f_ref.write(ref + "\n")
    command = f"python3 -m bleurt.score_files -candidate_file=_temp.mt -reference_file=_temp.ref -bleurt_checkpoint={bleurt_path} -scores_file=_temp.out"
    subprocess.run(command, shell=True, check=True)

    with open("_temp.out", "r") as f_out:
        scores = [float(line.strip()) for line in f_out]

    os.remove("_temp.mt")
    os.remove("_temp.ref")
    os.remove("_temp.out")

    return {"scores": scores, "system_score": sum(scores) / len(scores)}




def main(mt_path: str, ref_path: str, output_path: str,bleurt_path: str = "BLEURT-20") -> Dict:

    with open(mt_path, "r") as f_mt, open(ref_path, "r") as f_ref:
        mt_list = [line.strip() for line in f_mt]
        ref_list = [line.strip() for line in f_ref]
        
    results = func_call(bleurt_path, mt_list, ref_list)

    with open(output_path, "w") as f_out:
        for score in results["scores"]:
            f_out.write(f"{score}\n")


if __name__ == "__main__":
    fire.Fire(main)