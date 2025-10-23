from typing import Sequence, Dict

import torch


def data_collator_cd(instances: Sequence[Dict]) -> Dict:
    pixel_values_list = [item["pixel_values"] for item in instances]
    if len(pixel_values_list[0].shape) == 3:
        # training with crops: (B, 6, H, W)
        pixel_values = torch.stack(pixel_values_list, dim=0)
    elif len(pixel_values_list[0].shape) == 4:
        # testing with sliding windows: (NxB, 6, H, W)
        pixel_values = torch.cat(pixel_values_list, dim=0)
    else:
        raise RuntimeError

    labels_list = [item["labels"] for item in instances]
    if len(labels_list[0].shape) == 2:
        labels = torch.stack(labels_list, dim=0)  # training (B, H, W)
    elif len(labels_list[0].shape) == 3:
        labels = torch.cat(labels_list, dim=0)  # testing (NxB, H, W)
    else:
        raise RuntimeError

    batch = dict(pixel_values=pixel_values, labels=labels)

    if "sliding_boxes" in instances[0]:
        # (NxB, 4)
        batch["sliding_boxes"] = torch.cat([item["sliding_boxes"] for item in instances], dim=0)
    if "input_size" in instances[0]:
        # (NxB, 2)
        batch["input_size"] = torch.cat([item["input_size"] for item in instances], dim=0)
    if "id" in instances[0]:
        # (NxB,)
        batch["id"] = []
        for item in instances:
            batch["id"].extend(item["id"])

    return batch