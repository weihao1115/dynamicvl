import argparse
import json
import os
import string
from os.path import join, dirname

import numpy as np
import dvl.vqa

from dvl.vqa.pretty_print import dict_to_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default=f"{dirname(dvl.__file__)}/../../results/vqa")
    args = parser.parse_args()

    model_list = os.listdir(args.result_dir)
    model_list.sort()

    result_libs = {}
    for task in ["BCA-QA", "CSE-QA", "EA"]:
        print(f"Processing task: {task}")

        for model in model_list:
            model_result_dir = join(args.result_dir, model)
            if not os.path.isdir(model_result_dir):
                continue

            result_name = model
            if result_name not in result_libs:
                result_libs[result_name] = []

            result_json_path = f"{model_result_dir}/{task}.json"
            if not os.path.exists(result_json_path):
                print(f"[WARN]: {result_name} {task} does not exist!")  # 添加这行
                result_libs[result_name].extend([np.nan] if task in ["EA"] else [np.nan, np.nan])
                continue

            with open(result_json_path, "r") as f:
                result_data = json.load(f)

            if task in ["BCA-QA", "CSE-QA"]:
                splits = ["Single choice", "Multiple choice"]
            else:
                splits = ["Single choice"]

            for split in splits:
                acc_list = []
                for doc in result_data:
                    response = doc["response"]
                    task_type = split
                    metadata = doc["metadata"]

                    if task in ["BCA-QA", "CSE-QA"] and metadata["task_type"] != split:
                        continue

                    gt = metadata["ground_truth_option"].strip(".")
                    pred = response.strip(".")

                    if task_type == "Single choice":
                        pred = pred.strip().lower()
                        gt = gt.strip().lower()
                        if pred == gt:
                            acc_list.append(1)
                        else:
                            acc_list.append(0)

                    elif task_type == "Multiple choice":
                        pred = pred.strip().lower().split(",")
                        pred = [item.strip().strip(".") for item in pred]
                        pred = sorted([item for item in pred if item in string.ascii_lowercase])

                        gt = gt.strip().lower().split(",")
                        gt = sorted([item.strip().strip(".") for item in gt])

                        if pred == gt:
                            acc_list.append(1)
                        else:
                            acc_list.append(0)

                    else:
                        raise ValueError("Unknown task type")

                if len(acc_list) > 0:
                    accuracy = sum(acc_list) / len(acc_list)
                    result_libs[result_name].append(accuracy)
                else:
                    print(f"Warning: {result_name} {task} {split} has no valid data")
                    result_libs[result_name].append(np.nan)

    total_score_dict = {}
    for result_name in result_libs:
        total_score_dict[result_name] = [result_name]
        for acc_value in result_libs[result_name]:
            if isinstance(acc_value, float) and not np.isnan(acc_value):
                total_score_dict[result_name].append(f"{acc_value:.1%}")
            else:
                total_score_dict[result_name].append("N/A")

    custom_headers = [
        "index", "method", "BCA-QA (single)", "BCA-QA (multi)", "CSE-QA (single)", "CSE-QA (multi)", "EA"
    ]
    total_score_dict = {idx: value for idx, value in enumerate(total_score_dict.values())}
    table = dict_to_table(total_score_dict, custom_headers)
    print(table)

    with open(f"{args.result_dir}/accuracy_table.txt", "w") as f:
        f.write(str(table))


if __name__ == '__main__':
    main()

