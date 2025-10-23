import argparse
import glob
import json
import logging
import os
from os.path import join, basename

import yaml
import torch
from safetensors.torch import load_file
from transformers import Trainer

from dvl.cd.data_collators import data_collator_cd
from dvl.cd.dataset import DynamicVLChangeDetection
from dvl.cd.models import model_libs
from dvl.cd.train_cd import compute_metrics, eval_meter, acc_meter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=str, required=True)
    args = parser.parse_args()

    exp_path = args.exp_path
    if not exp_path.startswith("/"):
        exp_path = f"./results/{exp_path}"

    logging.info(f"Testing experiment: {exp_path}")

    ckpt_list = sorted(glob.glob(join(exp_path, "checkpoint-*")))
    if len(ckpt_list) == 0:
        logging.warning(f"No checkpoints found in {exp_path}")
        return

    logging.info(f"Found {len(ckpt_list)} checkpoint(s)")

    for idx, ckpt_path in enumerate(ckpt_list, 1):
        ckpt_name = basename(ckpt_path)
        test_metrics_path = join(ckpt_path, "test_metrics.json")

        if os.path.exists(test_metrics_path):
            logging.info(f"[{idx}/{len(ckpt_list)}] Skipping {ckpt_name} (already tested)")
            continue

        logging.info(f"[{idx}/{len(ckpt_list)}] Testing {ckpt_name}")

        # Ensure meter is clean before each checkpoint evaluation
        eval_meter.reset()
        acc_meter.reset()

        exp_config_path = join(exp_path, "config.yaml")
        with open(exp_config_path, 'r') as f:
            exp_config = yaml.safe_load(f)

        model_name = exp_config["model_name"]
        model_kwargs = exp_config["model_kwargs"]
        model_cls = model_libs[model_name]
        model = model_cls(**model_kwargs)
        model.eval()
        model.main_input_name = "pixel_values"  # for include_for_metrics=["inputs"]

        logging.info(f"  Loading model weights...")
        model_safetensors_path = join(ckpt_path, "model.safetensors")
        state = load_file(model_safetensors_path)
        if any(k.startswith("module.") for k in state.keys()):
            state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        model.to("cuda")

        eval_dataset_kwargs = exp_config.get("eval_dataset_kwargs", {})
        eval_dataset = DynamicVLChangeDetection(split="test", **eval_dataset_kwargs)
        logging.info(f"  Test dataset size: {len(eval_dataset)}")

        training_args = torch.load(f"{ckpt_path}/training_args.bin", weights_only=False)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=None,
            eval_dataset=eval_dataset,
            data_collator=data_collator_cd,
            compute_metrics=compute_metrics,
        )

        logging.info(f"  Running evaluation...")
        metrics = trainer.evaluate()

        with open(test_metrics_path, "w") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

        logging.info(f"  mIoU: {metrics.get('mIoU', 0):.4f} | mF1: {metrics.get('F1', 0):.4f}")
        logging.info(f"  Saved to {test_metrics_path}")

    logging.info("Testing completed")


if __name__ == '__main__':
    main()
