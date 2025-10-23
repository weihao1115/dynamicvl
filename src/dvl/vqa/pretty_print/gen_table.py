import argparse
import json
import os

from os.path import join, dirname

import numpy as np
import dvl.vqa

from dvl.vqa.pretty_print import dict_to_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default=f"{dirname(dvl.__file__)}/../../results/vqa")
    parser.add_argument('--gpt_model_id', type=str, default="gpt-4.1-mini")
    args = parser.parse_args()

    model_list = os.listdir(args.result_dir)
    model_list.sort()

    result_libs = {}
    for task in ["BCA-Report", "CSE-Report", "RCC", "DTC"]:
        print(f"[INFO] Processing task: {task}")

        for model in model_list:
            model_result_dir = join(args.result_dir, model)
            if not os.path.isdir(model_result_dir):
                continue

            result_name = model
            if result_name not in result_libs:
                result_libs[result_name] = []

            result_jsonl_path = f"{model_result_dir}/{task}.{args.gpt_model_id}.jsonl"
            if not os.path.exists(result_jsonl_path):
                print(f"[WARN] {result_name} {task} does not exist!")
                result_libs[result_name].extend(["N/A"] * 5 if task in ["RCC"] else ["N/A"] * 4)
                if task != "DTC":
                    result_libs[result_name].append("")
                continue

            with open(result_jsonl_path, "r") as f:
                result_data = [json.loads(line.strip()) for line in f.readlines()]

            if task == "BCA-Report":
                score_types = ["land_cover_type_identification_accuracy", "time_period_accuracy", "change_quantification_accuracy"]
            elif task == "CSE-Report":
                score_types = ["change_rate_precision", "time_period_accuracy", "change_pattern_accuracy"]
            elif task == "RCC":
                score_types = ["temporal_coverage", "spatial_accuracy", "process_fidelity", "region_containment"]
            elif task == "DTC":
                score_types = ["temporal_coverage", "spatial_accuracy", "process_fidelity"]
            else:
                raise NotImplementedError

            score_dict = {}
            for doc in result_data:
                response = doc["response"]
                for key in score_types:
                    prefix_tag = f"<{key}>"
                    suffix_tag = f"</{key}>"
                    key_score = 0.0
                    if response is not None:
                        if prefix_tag in response and suffix_tag in response:
                            key_score = response.split(prefix_tag)[1].split(suffix_tag)[0].strip()
                        elif prefix_tag in response:
                            key_score = response.split(prefix_tag)[1][0].strip()
                        elif suffix_tag in response:
                            key_score = response.split(suffix_tag)[0][-1].strip()
                        try:
                            key_score = int(key_score)
                        except:
                            key_score = 0.0

                    if key not in score_dict:
                        score_dict[key] = []
                    score_dict[key].append(key_score)

            key_avg_scores = {}
            for key in score_types:
                try:
                    key_avg_scores[key] = sum(score_dict[key]) / len(score_dict[key])
                except Exception as e:
                    print(f"[WARN] Error calculating {key} avg for {result_name} {task}: {e}")
                    key_avg_scores[key] = None

            try:
                total_avg_score = sum(key_avg_scores.values()) / len(key_avg_scores)
            except Exception as e:
                print(f"[WARN] Error calculating avg for {result_name} {task}: {e}")
                total_avg_score = None

            result_libs[result_name].append("error" if total_avg_score is None else f"{total_avg_score:.2f}")
            for key in score_types:
                result_libs[result_name].append(
                    "error" if key_avg_scores[key] is None else f"{key_avg_scores[key]:.2f}"
                )

            if task != "DTC":
                result_libs[result_name].append("")

    total_score_dict = {}
    for result_name in result_libs:
        total_score_dict[result_name] = [result_name, *result_libs[result_name]]

    custom_headers = [
        "index", "modal",
        "BCA-AVG", "BCA-LCT", "BCA-TPA", "BCA-CQA",
        "|",
        "CSE-AVG", "CSE-CRP", "CSE-TPA", "CSE-CPA",
        "||",
        "RCC-AVG", "RCC-TC", "RCC-SA", "RCC-PF", "RCC-RC",
        "|||",
        "DTC-AVG", "DTC-TC", "DTC-SA", "DTC-PF",
    ]
    total_score_dict = {idx: value for idx, value in enumerate(total_score_dict.values())}
    table = dict_to_table(total_score_dict, custom_headers)
    print(table)

    with open(f"{args.result_dir}/generation_table.{args.gpt_model_id}.txt", "w") as f:
        f.write(str(table))


if __name__ == '__main__':
    main()

