import json
from os.path import join
from typing import Dict, Optional

import albumentations
import albumentations.pytorch
import numpy as np
import torch
from PIL import Image
from albumentations import Compose

from torch.utils.data import Dataset

from dvl.cd.data_collators import data_collator_cd
from dvl.utils.data import sliding_window


LABEL_TO_SEMANTIC_NAME = {
    1: 'vegetation',
    2: 'non vegetated surface',
    3: 'water',
    4: 'building',
    5: 'playground'
}


class DynamicVLChangeDetection(Dataset):
    def __init__(
            self,
            split: str,
            transforms: Dict,
            data_dir: str,
            pair_mode: str,
            filter_zero_mask: bool = True,
            sliding_kernel: Optional[int] = None,
            sliding_stride: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.split = split

        if sliding_kernel is not None:
            assert split == "test", split
            if sliding_stride is None:
                sliding_stride = sliding_kernel // 2

        self.sliding_kernel = sliding_kernel
        self.sliding_stride = sliding_stride

        if pair_mode not in ["endpoints", "adjacent"]:
            raise ValueError("pair_mode must be either 'endpoints' or 'adjacent'")

        with open(join(self.data_dir, f"cd_{pair_mode}.json"), "r") as f:
            data = json.load(f)

        # filter the masks without any changes
        filtered_num = 0
        self.data = []
        for doc in data:
            if filter_zero_mask:
                gt_mask_path = doc["gt_mask_path"]
                gt_mask = np.array(Image.open(gt_mask_path).convert("L"), dtype=np.uint8)
                if gt_mask.max() == 0 and gt_mask.min() == 0:
                    filtered_num += 1
                    continue
            self.data.append(doc)

        if filtered_num > 0:
            print(f"[WARN] filter out {filtered_num} samples without any changes from {split} split!")

        # assert "Normalize" in transforms, transforms
        assert "ToTensorV2" in transforms, transforms
        transform_list = []
        for key, value in transforms.items():
            try:
                transform_cls = getattr(albumentations, key)
            except AttributeError:
                transform_cls = getattr(albumentations.pytorch, key)
            transform_list.append(transform_cls(**value))
        self.transforms = Compose(transforms=transform_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_sample = self.data[idx]
        data_id = data_sample["id"]
        t1_image_path = join(self.data_dir, self.split, data_sample["t1_image_path"])
        t2_image_path = join(self.data_dir, self.split, data_sample["t2_image_path"])
        gt_mask_path = join(self.data_dir, self.split, data_sample["gt_mask_path"])

        t1_image = np.array(Image.open(t1_image_path).convert("RGB"), dtype=np.uint8)
        t2_image = np.array(Image.open(t2_image_path).convert("RGB"), dtype=np.uint8)
        # 2 x (H, W, 3) -> (H, W, 6)
        pixel_values = np.concatenate((t1_image, t2_image), axis=2)
        # (H, W)
        labels = np.array(Image.open(gt_mask_path).convert("L"), dtype=np.uint8)

        src_labels = labels // 10
        tgt_labels = labels % 10
        assert np.sum(src_labels >= 6) == 0
        assert np.sum(tgt_labels >= 6) == 0
        # {0, 11, ..., 55} -> 0-35
        labels = src_labels * 6 + tgt_labels

        blob = self.transforms(image=pixel_values, mask=labels)
        pixel_values = blob["image"]
        labels = blob["mask"]

        if self.sliding_kernel is not None:
            input_size = pixel_values.shape[-2:]
            # (N, 4) np.array
            sliding_boxes = sliding_window(input_size=input_size, kernel_size=self.sliding_kernel, stride=self.sliding_stride)
            rgb_patches = []
            for box in sliding_boxes:
                xmin, ymin, xmax, ymax = box
                rgb_patches.append(pixel_values[:, ymin:ymax, xmin:xmax])

            # (N, 6, H, W)
            pixel_values = torch.stack(rgb_patches, dim=0)
            # (N, 4) torch.tensor
            sliding_boxes = torch.from_numpy(sliding_boxes)
            # (H, W) -> (N, H, W)
            labels = labels.unsqueeze(0).expand(pixel_values.shape[0], *labels.shape)
            # (2,) -> (N, 2)
            input_size = torch.tensor([input_size for _ in range(pixel_values.shape[0])])
            # (N,)
            data_id = [data_id for _ in range(pixel_values.shape[0])]
            return dict(
                pixel_values=pixel_values,
                labels=labels,
                sliding_boxes=sliding_boxes,
                input_size=input_size,
                id=data_id,
            )

        else:
            return dict(
                pixel_values=pixel_values,
                labels=labels,
            )


if __name__ == '__main__':
    dataset = DynamicVLChangeDetection(
        data_dir="data/s2_wcd",
        split="train",
        transforms={
            "PadIfNeeded": dict(min_height=518, min_width=518, border_mode=0, value=0, p=1.0),
            "RandomCrop": dict(height=518, width=518, p=1.0),
            "HorizontalFlip": dict(p=0.5),
            "VerticalFlip": dict(p=0.5),
            "RandomRotate90": dict(p=1.0),
            "Normalize": dict(
                mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225],
                max_pixel_value=1.0,
                always_apply=True
            ),
            "ToTensorV2": dict(transpose_mask=True),
        }
    )
    batch = data_collator_cd([dataset[i] for i in range(3)])


    dataset = DynamicVLChangeDetection(
        data_dir="data/s2_wcd",
        split="test",
        sliding_kernel=518,
        transforms={
            "Normalize": dict(
                mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225],
                max_pixel_value=1.0,
                always_apply=True
            ),
            "ToTensorV2": dict(transpose_mask=True),
        }
    )
    batch = data_collator_cd([dataset[i] for i in range(3)])


