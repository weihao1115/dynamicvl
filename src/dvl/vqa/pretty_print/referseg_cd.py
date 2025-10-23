import argparse
import copy
import glob
import json
import os
from os.path import dirname

from PIL import Image
from tqdm import tqdm

from typing import Dict, List, Union, Any

import math
import numpy as np
import torch

import dvl
from dvl.vqa.dataset import DynamicVLReferSeg


def to_cpu(input_tensor: torch.Tensor):
    if hasattr(input_tensor, 'detach'):
        input_tensor = input_tensor.detach()
    if hasattr(input_tensor, 'cpu'):
        input_tensor = input_tensor.cpu()
    return input_tensor


def to_numpy(input_tensor: torch.Tensor):
    if isinstance(input_tensor, np.ndarray):
        return input_tensor
    input_tensor = to_cpu(input_tensor)
    return input_tensor.numpy()


def label_pred_true_preprocess(
        label_preds: Union[torch.Tensor, np.ndarray, List[torch.Tensor or np.ndarray]],
        label_trues: Union[torch.Tensor, np.ndarray, List[torch.Tensor or np.ndarray]]
) -> (List[np.ndarray], List[np.ndarray]):

    def flatten_labels(labels):
        labels = copy.deepcopy(labels) # out-place operation
        if isinstance(labels, torch.Tensor):
            labels = to_numpy(labels)
        if not isinstance(labels, list):
            if len(labels.shape) in [1, 2]:
                labels = [labels]
            elif len(labels.shape) == 3:
                labels = [i for i in labels]
            else:
                raise RuntimeError("Unexpected label shape")

        for i in range(len(labels)):
            # from torch.Tensor to np.ndarray
            if not isinstance(labels[i], np.ndarray):
                labels[i] = to_numpy(labels[i])
            # ensure the mask is a 1d vector with int-type
            labels[i] = labels[i].reshape(-1).astype(int)
        return labels

    label_preds = flatten_labels(label_preds)
    label_trues = flatten_labels(label_trues)
    for pred, label in zip(label_preds, label_trues):
        if len(pred) != len(label):
            raise RuntimeError(
                "Predicted masks and ground-truth labels have different lengths!"
                f"Got masks are {len(pred)} while labels are {len(label)}!"
            )
    return label_preds, label_trues


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def get_hist(image, label, num_class):
    hist = np.zeros((num_class, num_class))
    hist += fast_hist(image.flatten(), label.flatten(), num_class)
    return hist


def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa


def accuracy(pred, label, ignore_zero=False):
    valid = (label >= 0)
    if ignore_zero: valid = (label > 0)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


class MambaSCDEvaluator:
    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.reset()

    def update(
            self,
            label_trues: Union[torch.Tensor, np.ndarray, List[torch.Tensor or np.ndarray]],
            label_preds: Union[torch.Tensor, np.ndarray, List[torch.Tensor or np.ndarray]],
            index_name: Union[List[Any], Any]
    ):
        label_preds, label_trues = label_pred_true_preprocess(label_preds, label_trues)
        if not isinstance(index_name, List):
            index_name = [index_name]
        for i, (lt, lp) in enumerate(zip(label_trues, label_preds)):
            acc, _ = accuracy(pred=lp, label=lt)
            self.acc_meter.update(acc)

            index_hist = get_hist(image=lp, label=lt, num_class=self.n_classes)
            self.hist += index_hist
            self.index_results[index_name[i]] = self.compute(hist=index_hist)[0]

    def compute(self, hist=None) -> (Dict, Dict):
        if hist is None:
            hist = self.hist

        hist_fg = hist[1:, 1:]
        c2hist = np.zeros((2, 2))
        c2hist[0][0] = hist[0][0]
        c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
        c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
        c2hist[1][1] = hist_fg.sum()
        hist_n0 = hist.copy()
        hist_n0[0][0] = 0
        kappa_n0 = cal_kappa(hist_n0)
        iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
        IoU_fg = iu[1]
        IoU_mean = (iu[0] + iu[1]) / 2
        Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e

        pixel_sum = hist.sum()
        change_pred_sum = pixel_sum - hist.sum(1)[0].sum()
        SC_TP = np.diag(hist[1:, 1:]).sum()
        SC_Precision = SC_TP / (change_pred_sum + 1e-10)
        SC_Recall = SC_TP / (change_pred_sum + 1e-10)

        results_dict = {
            "kappa_n0": kappa_n0,
            "IoU_mean": IoU_mean,
            "IoU_fg": IoU_fg,
            "Sek": Sek,
            "Acc": self.acc_meter.avg,
            "Precision": SC_Precision,
            "Recall": SC_Recall,
        }
        return results_dict, self.index_results

    def reset(self):
        self.hist = np.zeros((self.n_classes, self.n_classes))
        self.acc_meter = AverageMeter()
        self.index_results = {}
        self.has_sync = False


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, count, weight):
        self.val = val
        self.avg = val
        self.count = count
        self.sum = val * weight
        self.initialized = True

    def update(self, val, count=1, weight=1):
        if not self.initialized:
            self.initialize(val, count, weight)
        else:
            self.add(val, count, weight)

    def add(self, val, count, weight):
        self.val = val
        self.count += count
        self.sum += val * weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True)
    args = parser.parse_args()

    score_path = f"{args.pred_dir}/cd_scores.json"
    if os.path.exists(score_path):
        with open(score_path, "r") as f:
            score_dict = json.load(f)
    else:
        evaluator = MambaSCDEvaluator(n_classes=5 * 5 + 1)
        dataset = DynamicVLReferSeg(data_dir=f"{dirname(dvl.__file__)}/../../data/test")

        for doc in tqdm(dataset):
            doc_id = doc["metadata"]["id"]
            gt_mask = doc["gt_mask"]
            gt_mask = torch.from_numpy(gt_mask)

            pred_mask_path = f"{args.pred_dir}/{doc_id}.png"
            if not os.path.exists(pred_mask_path):
                print(f"[ERROR] {doc_id}.png does not exist in {args.pred_dir}! Skip it")
                continue
            pred_mask = Image.open(pred_mask_path).convert("L")
            pred_mask = torch.from_numpy(np.array(pred_mask))
            pred_mask = (pred_mask > 0).int()

            src_idx, tgt_idx = doc["cd_info"]["src_idx"], doc["cd_info"]["tgt_idx"]
            sem_idx = (src_idx - 1) * 5 + tgt_idx

            gt_mask = gt_mask * sem_idx
            pred_mask = pred_mask * sem_idx
            evaluator.update(label_trues=gt_mask, label_preds=pred_mask, index_name=[doc_id])

        score_dict, _ = evaluator.compute()
        with open(score_path, "w") as f:
            json.dump(score_dict, f, indent=4)
        print(f"[INFO] overall CD metrics has been saved to {score_path}")

    print(
        f"[INFO] "
        f"OA={score_dict['Acc']:.2%}, "
        f"IoU-foreground={score_dict['IoU_fg']:.2%}, "
        f"SeK={score_dict['Sek']:.4f}, "
        f"Precision={score_dict['Precision']:.2%}, "
        f"Recall={score_dict['Recall']:.2%}, "
    )
