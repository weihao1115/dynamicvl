import argparse
import json
import os
from os.path import dirname

import numpy as np
from enum import Enum
import torch
from PIL import Image
from tqdm import tqdm

import dvl
from dvl.vqa.dataset import DynamicVLReferSeg


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape, f"{output.shape} != {target.shape}"
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True)
    args = parser.parse_args()

    score_path = f"{args.pred_dir}/iou_scores.json"
    if os.path.exists(score_path):
        with open(score_path, "r") as f:
            score_data = json.load(f)
        giou = score_data["giou"]
        ciou = score_data["ciou"]

    else:
        dataset = DynamicVLReferSeg(data_dir=f"{dirname(dvl.__file__)}/../../data/test")

        sample_scores = {}
        intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
        union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
        acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

        for doc in tqdm(dataset):
            doc_id = doc["metadata"]["id"]
            gt_mask = torch.from_numpy(doc["gt_mask"])

            pred_mask_path = f"{args.pred_dir}/{doc_id}.png"
            if not os.path.exists(pred_mask_path):
                print(f"[ERROR] {doc_id}.png does not exist in {args.pred_dir}! Skip it")
                continue
            pred_mask = Image.open(pred_mask_path).convert("L")
            pred_mask = torch.from_numpy(np.array(pred_mask))
            pred_mask = (pred_mask > 0).int()

            intersection, union, acc_iou = 0.0, 0.0, 0.0
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output=pred_mask.contiguous().clone().cuda(),
                target=gt_mask.contiguous().cuda(),
                K=2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target

            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = acc_iou.cpu().numpy()
            sample_scores[doc_id] = acc_iou[1].item()

            intersection_meter.update(intersection)
            union_meter.update(union)
            acc_iou_meter.update(acc_iou, n=1)

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        ciou = iou_class[1].item()
        giou = acc_iou_meter.avg[1].item()

        with open(score_path, "w") as f:
            json.dump(dict(giou=giou, ciou=ciou), f, indent=4)
        print(f"[INFO] overall IoU has been saved to {score_path}")

    print(f"[INFO] giou={giou:.2%}, ciou={ciou:.2%}")
