from typing import Literal, Dict, Any, List
import json
import os
from utils.configs import valid_langs
import subprocess
import fire
import eval.comet_eval as comet_eval
import eval.bleurt_eval as bleurt_eval
import eval.sacrebleu_eval as sacrebleu_eval
from collections import defaultdict



def run_vllm_inference(model_path: str, infer_data_path: str):
    script_name = "inference/run_vllm_mt_generation.py"
    generation_command = f"python3 {script_name} --model_path {model_path} --infer_data_path {infer_data_path}"
    subprocess.run(generation_command, shell=True, check=True)


def flat_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = text.strip()
    return text


def _group_data_aggregation(grouped_data: dict, metrics: list[str]) -> dict:
    lang_pair_scores = {}
    for pair, data in grouped_data.items():
        lang_pair_scores[pair] = {}
        src_lang, trg_lang = pair

        # Corpus-level SacreBLEU
        if "sacrebleu" in metrics:
            sacrebleu_out = sacrebleu_eval.func_call(
                data["mt"], data["ref"], trg_lang=trg_lang
            )
            lang_pair_scores[pair]["sacrebleu"] = sacrebleu_out["system_score"]

        # Average of sentence-level BLEURT
        if "bleurt" in metrics:
            if data["bleurt_scores"]:
                avg_bleurt = sum(data["bleurt_scores"]) / len(data["bleurt_scores"])
                lang_pair_scores[pair]["bleurt"] = avg_bleurt
        # Average of sentence-level COMET
        if "comet" in metrics:
            if data["comet_scores"]:
                avg_comet = sum(data["comet_scores"]) / len(data["comet_scores"])
                lang_pair_scores[pair]["comet"] = avg_comet
    return lang_pair_scores


def run_evaluation(infer_data_path: str, metrics: list[str], **kwargs):
    """
    Runs evaluation on inference data, calculating metrics at the corpus level
    for each language pair.
    """
    with open(infer_data_path, "r") as f:
        infer_data = json.load(f)

    # --- Step 1: Pre-calculate sentence-level scores (BLEURT) if needed ---
    # BLEURT is still calculated at the sentence level. We do it once for all
    # clips to avoid the overhead of loading the model multiple times.
    if "bleurt" in metrics:
        full_mt_list = [flat_text(item["mt_text"]) for item in infer_data]
        full_ref_list = [flat_text(item["trg_text"]) for item in infer_data]

        bleurt_path = kwargs.get("bleurt_path", None)
        if bleurt_path is None:
            raise ValueError("BLEURT path must be provided when 'bleurt' is in metrics.")

        # use cli call to avoid GPU memory occupation
        bleurt_out = bleurt_eval.cli_call(bleurt_path, full_mt_list, full_ref_list)
        if len(bleurt_out["scores"]) != len(infer_data):
            raise ValueError(
                f"BLEURT output scores length ({len(bleurt_out['scores'])}) does not "
                f"match the number of items ({len(infer_data)})"
            )
        # Attach scores to each item for later grouping and averaging.
        for item, score in zip(infer_data, bleurt_out["scores"]):
            item["bleurt"] = score

    if "comet" in metrics:
        full_src_list = [flat_text(item["src_text"]) for item in infer_data]
        full_mt_list = [flat_text(item["mt_text"]) for item in infer_data]
        full_ref_list = [flat_text(item["trg_text"]) for item in infer_data]

        comet_path = kwargs.get("comet_path", None)
        if comet_path is None:
            raise ValueError("COMET path must be provided when 'comet' is in metrics.")
        
        comet_out = comet_eval.func_call(comet_path, full_src_list, full_mt_list, full_ref_list)
        if len(comet_out["scores"]) != len(infer_data):
            raise ValueError(
                f"COMET output scores length ({len(comet_out['scores'])}) does not "
                f"match the number of items ({len(infer_data)})"
            )
        # Attach scores to each item for later grouping and averaging.
        for item, score in zip(infer_data, comet_out["scores"]):
            item["comet"] = score

    # --- Step 2: Group clips by language pair ---
    grouped_data = defaultdict(lambda: {"mt": [], "ref": [], "bleurt_scores": [], "comet_scores": []})

    for item in infer_data:
        pair = (item["src_lang"], item["trg_lang"])
        grouped_data[pair]["mt"].append(flat_text(item["mt_text"]))
        grouped_data[pair]["ref"].append(flat_text(item["trg_text"]))
        if "bleurt" in metrics and "bleurt" in item:
            grouped_data[pair]["bleurt_scores"].append(item["bleurt"])
        if "comet" in metrics and "comet" in item:
            grouped_data[pair]["comet_scores"].append(item["comet"])

    # --- Step 3: Calculate corpus-level scores for each group ---
    lang_pair_scores = _group_data_aggregation(grouped_data, metrics)

    return lang_pair_scores


def run(
    model_path: str,
    infer_data_path: str,
    metrics: list[str] = ["bleurt", "sacrebleu", "comet"],
    bleurt_path: str = "BLEURT-20",
    comet_path: str = "wmt22-comet-da/checkpoints/model.ckpt",
    log_to_wandb: bool = True,
    config: dict = {},
):
    if isinstance(metrics, str):
        metrics = metrics.split(",")
    
    if "model_name" not in config:
        config["model_name"] = os.path.basename(model_path)

    # check infer data
    if not os.path.exists(infer_data_path):
        raise ValueError(f"Inference data path {infer_data_path} does not exist.")
    
    check_fields = ["src_lang", "trg_lang", "src_text", "trg_text"]
    with open(infer_data_path, "r") as f:
        infer_data = json.load(f)
    for data_item in infer_data:
        if not all(field in data_item for field in check_fields):
            raise ValueError(f"Missing required fields {check_fields} in inference data item: {data_item}")
    
    print("Running model inference...")
    run_vllm_inference(model_path, infer_data_path)

    print("Running evaluation...")
    lang_pair_scores = run_evaluation(infer_data_path, metrics, bleurt_path=bleurt_path, comet_path=comet_path)

    print("Logging results...")
    log_results(lang_pair_scores, config, metrics, log_to_wandb=log_to_wandb)
        
    if log_to_wandb:
        print(f"Finished logging results for {config['model_name']} to wandb.")
    else:
        print(f"Finished logging results for {config['model_name']}.")

def log_results(
    lang_pair_scores: Dict[tuple, Dict[str, float]],
    config: Dict[str, Any],
    metrics: List[str],
    log_to_wandb: bool = True,
    valid_langs: List[str] = valid_langs,
):
    """
    Logs evaluation metrics to wandb and/or stdout.

    Args:
        lang_pair_scores: A dictionary with evaluation scores for each language pair.
        config: A dictionary containing the configuration for the run.
        metrics: A list of metrics that were calculated.
        log_to_wandb: Whether to log to wandb or just print to stdout.
        valid_langs: List of valid language codes.
    """
    
    # Calculate overall scores first (used by both logging methods)
    overall_scores = {}
    for metric in metrics:
        # Collect the scores from all valid language pairs
        metric_scores_across_pairs = [
            scores[metric]
            for pair, scores in lang_pair_scores.items()
            if metric in scores and scores[metric] is not None
        ]

        # Calculate the final average if any scores were found
        if metric_scores_across_pairs:
            overall_scores[metric] = sum(metric_scores_across_pairs) / len(
                metric_scores_across_pairs
            )

    if log_to_wandb:
        log_results_to_wandb(lang_pair_scores, config, metrics, valid_langs, overall_scores)
    else:
        log_results_to_stdout(lang_pair_scores, config, metrics, valid_langs, overall_scores)

def log_results_to_wandb(
    lang_pair_scores: Dict[tuple, Dict[str, float]],
    config: Dict[str, Any],
    metrics: List[str],
    valid_langs: List[str],
    overall_scores: Dict[str, float],
):
    """
    Initializes a wandb run and logs evaluation metrics, tables, and matrices.
    """
    import wandb

    project_name = "eax-eval-matrix"
    wandb.init(
        project=project_name,
        name=f"{config['model_name']}",
        config=config,
    )

    # 1. Log score matrices (or lists for transcribe) to wandb
    for metric in metrics:
        # Check if we have any scores for this metric before proceeding
        if not any(
            metric in pair_scores for pair_scores in lang_pair_scores.values()
        ):
            continue

        columns = ["source_lang \\ target_lang"] + valid_langs
        score_matrix_data = []
        for source_lang in valid_langs:
            row: list[Any] = [source_lang]
            for target_lang in valid_langs:
                # Retrieve the corpus-level score directly
                score = lang_pair_scores.get((source_lang, target_lang), {}).get(metric, None)
                row.append(round(score, 4) if isinstance(score, float) else score)
            score_matrix_data.append(row)

        table = wandb.Table(columns=columns, data=score_matrix_data)
        wandb.log({f"score_matrix/{metric}": table})

    # 2. Report overall performance table
    if overall_scores:
        overall_table_data = [
            [metric, round(score, 4)] for metric, score in overall_scores.items()
        ]
        overall_table = wandb.Table(
            columns=["Metric", "Overall Score (avg over lang pairs)"],
            data=overall_table_data,
        )
        wandb.log({"overall_performance": overall_table})

    wandb.finish()

def log_results_to_stdout(
    lang_pair_scores: Dict[tuple, Dict[str, float]],
    config: Dict[str, Any],
    metrics: List[str],
    valid_langs: List[str],
    overall_scores: Dict[str, float],
):
    """
    Prints evaluation results to stdout in a formatted way.
    """
    print("\n" + "="*80)
    print(f"EVALUATION RESULTS: {config['model_name']}")
    print("="*80)
    
    # Print overall scores
    if overall_scores:
        print("\nOVERALL PERFORMANCE (average over language pairs):")
        print("-" * 50)
        for metric, score in overall_scores.items():
            print(f"{metric.upper():>12}: {score:.4f}")
    
    # Print detailed score matrices for each metric
    for metric in metrics:
        # Check if we have any scores for this metric
        if not any(
            metric in pair_scores for pair_scores in lang_pair_scores.values()
        ):
            continue
            
        print(f"\n{metric.upper()} SCORE MATRIX:")
        print("-" * 50)
        
        # Print header
        header = f"{'src & tgt':>8}"
        for target_lang in valid_langs:
            header += f"{target_lang:>8}"
        print(header)
        
        # Print scores for each source language
        for source_lang in valid_langs:
            row = f"{source_lang:>8}"
            for target_lang in valid_langs:
                score = lang_pair_scores.get((source_lang, target_lang), {}).get(metric, None)
                if isinstance(score, float):
                    row += f"{score:>8.4f}"
                elif score is None:
                    row += f"{'N/A':>8}"
                else:
                    row += f"{str(score):>8}"
            print(row)
    
    # Print language pair details
    print(f"\nDETAILED SCORES BY LANGUAGE PAIR:")
    print("-" * 50)
    for (source_lang, target_lang), scores in sorted(lang_pair_scores.items()):
        print(f"{source_lang} -> {target_lang}:")
        for metric, score in scores.items():
            if isinstance(score, float):
                print(f"  {metric:>12}: {score:.4f}")
            else:
                print(f"  {metric:>12}: {score}")
    
    print("\n" + "="*80)



if __name__ == "__main__":
    fire.Fire(run)