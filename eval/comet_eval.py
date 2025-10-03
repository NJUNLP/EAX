from comet import download_model, load_from_checkpoint
from typing import Dict
import fire


def func_call(comet_path , src_list, mt_list, ref_list, model=None):
    assert len(src_list) == len(mt_list) == len(ref_list)

    data = []
    for src, mt, ref in zip(src_list, mt_list, ref_list):
            data.append({"src": src.strip(), "mt": mt.strip(), "ref": ref.strip()})

    if model is None:
        model = load_from_checkpoint(comet_path)

    model_output = model.predict(data, batch_size=32, gpus=1)
    print ("comet: ", model_output.system_score)
    
    return {
        "scores": model_output.scores,
        "metadata": (
            model_output.metadata if hasattr(model_output, "metadata") else None
        ),
        "system_score": model_output.system_score,
    }


def main(comet_path: str, src_path: str, mt_path: str, ref_path: str) -> Dict:
    
    src_list = []
    mt_list = []
    ref_list = []


    with (
        open(src_path, "r") as f_src,
        open(mt_path, "r") as f_mt,
        open(ref_path, "r") as f_ref,
    ):
        for src, mt, ref in zip(f_src, f_mt, f_ref):
            src_list.append(src.strip())
            mt_list.append(mt.strip())
            ref_list.append(ref.strip())

    return func_call(comet_path, src_list, mt_list, ref_list)

if __name__ == "__main__":
    fire.Fire(main)