import argparse
import glob
import importlib
import math
import os
from os.path import relpath, join, dirname
from typing import Sequence

import torch
import yaml
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers.trainer_pt_utils import save_state

import dvl.cd
from dvl.cd.data_collators import data_collator_cd
from dvl.cd.dataset import DynamicVLChangeDetection
from dvl.cd.eval_metrics.acc import AverageMeter, accuracy
from dvl.cd.eval_metrics.misc import EvalMeter
from dvl.cd.models import model_libs
from dvl.utils.train import get_unique_job_id, is_global_zero


class LogAllLearningRatesCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and kwargs["optimizer"] is not None:
            for i, param_group in enumerate(kwargs["optimizer"].param_groups):
                logs[f"lr_{param_group.get('name', f'group_{i}')}"] = param_group['lr']


def merge_patches_in_sliding_window(input_size, logits, sliding_boxes):
    softmax_prob = torch.softmax(logits, dim=1)
    win_h, win_w = int(input_size[0]), int(input_size[1])
    batch_size, num_classes, patch_h, patch_w = softmax_prob.shape
    merged = torch.zeros((1, num_classes, win_h, win_w), device=softmax_prob.device)
    counts = torch.zeros((1, 1, win_h, win_w), device=softmax_prob.device)

    for patch_idx, box in enumerate(sliding_boxes):
        xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        merged[:, :, ymin: ymax, xmin: xmax] += softmax_prob[patch_idx]
        counts[:, :, ymin: ymax, xmin: xmax] += 1

    return merged / (counts + 1e-8)


eval_meter = EvalMeter(num_class=35)
acc_meter = AverageMeter()


def compute_metrics(eval_pred, compute_result: bool = False):
    if isinstance(eval_pred.predictions, Sequence):
        # mask2former case
        logits = eval_pred.predictions[-1]
    else:
        logits = eval_pred.predictions
    logits = torch.as_tensor(logits)  # ← 不是 eval_pred 本身
    labels = torch.as_tensor(eval_pred.label_ids).long()
    inputs = getattr(eval_pred, "inputs", None)
    assert inputs is not None

    if is_global_zero():
        unique_ids = list(set(inputs["id"]))
        for u_id in unique_ids:
            u_indis = [idx for idx, item in enumerate(inputs["id"]) if item == u_id]
            u_logits = logits[u_indis]
            u_sliding_boxes = inputs["sliding_boxes"][u_indis]
            u_label = labels[u_indis[0]]
            u_input_size = inputs["input_size"][u_indis[0]]

            merged_softmax_prob = merge_patches_in_sliding_window(
                input_size=u_input_size, logits=u_logits, sliding_boxes=u_sliding_boxes
            )

            preds = merged_softmax_prob.argmax(dim=1).cpu()
            refs = u_label.unsqueeze(0).cpu()
            eval_meter.update(preds, refs)

            acc, _ = accuracy(preds, refs)
            acc_meter.update(acc, count=preds.size(0))

        if compute_result:
            eval_metrics = eval_meter.compute()
            eval_meter.reset()

            total_acc = acc_meter.avg
            acc_meter.reset()

            return {
                "mIoU": eval_metrics["miou"],
                "KappaCoefficient": eval_metrics["kappa_n0"],
                "F1": eval_metrics["f1"],
                "SeK": eval_metrics["sek"],
                "OA": total_acc,
            }
    else:
        if compute_result:
            eval_meter.reset()
            acc_meter.reset()
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--result_save_dir', type=str, default=f"{dirname(dvl.cd.__file__)}/results")
    parser.add_argument('--output_dir', type=str, default=None, help="Used for resuming")
    parser.add_argument('--override', nargs='*', default=[], help="Override training_args with key value pairs, e.g., --override learning_rate 0.001 num_train_epochs 10")
    args = parser.parse_args()

    config_path = args.config
    if not config_path.startswith("/"):
        config_path = f"{dirname(dvl.cd.__file__)}/configs/{config_path}"
    if not config_path.endswith(".yaml"):
        config_path += ".yaml"

    if args.output_dir is None:
        config_rel_name = relpath(config_path, "./configs").replace(".yaml", "")
        job_id = get_unique_job_id()
        output_basename_w_id = config_rel_name.split('/')[-1]
        if job_id:
            output_basename_w_id = f"{job_id}_{output_basename_w_id}"

        config_rel_name = "/".join(config_rel_name.split("/")[:-1]) + "/" + output_basename_w_id
        output_dir = join(args.result_save_dir, config_rel_name)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = args.output_dir if args.output_dir.startswith("/") else join(args.result_save_dir, args.output_dir)

    training_args = dict(
        output_dir=output_dir,
        report_to=["none"],
        eval_strategy="epoch",
        save_strategy="best",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="mIoU",
        greater_is_better=True,
        remove_unused_columns=False,
        batch_eval_metrics=True,
        eval_accumulation_steps=1,
        include_for_metrics=["inputs"],
        eval_use_gather_object=True,
        logging_steps=1,
    )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(join(output_dir, "config.yaml"), 'w', encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)

    sche_name = config.get("sche_name")
    sche_src = config.get("sche_src", "torch.optim.lr_scheduler")
    sche_kwargs = config.get("sche_kwargs", {})
    sche_cls = getattr(importlib.import_module(sche_src), sche_name)

    # wrap up the training arguments before launching the trainer
    training_args.update(config["training_args"])

    # Apply command-line overrides
    if args.override:
        if len(args.override) % 2 != 0:
            raise ValueError("--override requires an even number of arguments (key-value pairs)")
        override_dict = {}
        for i in range(0, len(args.override), 2):
            key = args.override[i]
            value = args.override[i + 1]
            # Try to infer the type from the original value if it exists
            if key in training_args and training_args[key] is not None:
                original_type = type(training_args[key])
                if original_type == bool:
                    value = value.lower() in ['true', '1', 'yes']
                elif original_type in [int, float]:
                    value = original_type(value)
            else:
                # Try to auto-convert to appropriate type
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    if value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
            override_dict[key] = value
        training_args.update(override_dict)

    training_args = TrainingArguments(**training_args)

    with open(join(output_dir, "training_args.yaml"), 'w', encoding="utf-8") as f:
        training_args_dict = {key: str(value) for key, value in vars(training_args).items()}
        yaml.dump(training_args_dict, f, sort_keys=False)

    # build model by model_name & model_kwargs
    model_name = config.get("model_name")
    model_kwargs = config.get("model_kwargs", {})
    model_cls = model_libs[model_name]
    model = model_cls(**model_kwargs)
    model.main_input_name = "pixel_values"  # for include_for_metrics=["inputs"]
    with open(join(output_dir, "model.txt"), 'w') as f:
        f.write(str(model))

    # build train_dataset & eval_dataset & data_collator by data_name & data_kwargs
    train_dataset_kwargs = config.get("train_dataset_kwargs", {})
    train_dataset = DynamicVLChangeDetection(split="train", **train_dataset_kwargs)

    eval_dataset_kwargs = config.get("eval_dataset_kwargs", {})
    eval_dataset = DynamicVLChangeDetection(split="val", **eval_dataset_kwargs)

    # build optimizer by optim_name & optim_kwargs
    optim_name = config.get("optim_name", "AdamW")
    optim_kwargs = config.get("optim_kwargs", {})

    learning_rate = training_args.learning_rate
    enc_lr_weight = optim_kwargs.pop("enc_lr_weight", 1.0)

    # Only include parameters that require grad and are not complex/buffer types
    encoder_params = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    decoder_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]

    if len(encoder_params) == 0:
        print("[WARN] No params starting with `encoder.` found in your model! enc_lr_weight won't be effective.")

    params = [
        {
            'params': encoder_params,
            'lr': learning_rate * enc_lr_weight,
            'init_lr': learning_rate * enc_lr_weight,
            'name': 'encoder',
            **optim_kwargs
        },
        {
            'params': decoder_params,
            'lr': learning_rate,
            'init_lr': learning_rate,
            'name': 'decoder',
            **optim_kwargs
        }
    ]
    optimizer = getattr(torch.optim, optim_name)(params=params)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator_cd,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
        callbacks=[LogAllLearningRatesCallback()],
    )

    train_dl = trainer.get_train_dataloader()
    steps_per_epoch = math.ceil(len(train_dl) / training_args.gradient_accumulation_steps)
    total_iters = (training_args.max_steps
                   if training_args.max_steps > 0
                   else int(steps_per_epoch * training_args.num_train_epochs))

    if sche_src == "dvl.cd.lr_scheduler":
        if sche_name == "WarmUpPolyLR":
            scheduler = sche_cls(optimizer=trainer.optimizer, total_iters=total_iters, **sche_kwargs)
        else:
            raise NotImplementedError
    else:
        scheduler = sche_cls(optimizer=trainer.optimizer, **sche_kwargs)

    trainer.lr_scheduler = scheduler

    # Check for existing checkpoints
    ckpt_list = sorted(glob.glob(f"{output_dir}/checkpoint-*"))
    trainer.train(resume_from_checkpoint=len(ckpt_list) >= 1)
    save_state(trainer)

if __name__ == '__main__':
    main()

