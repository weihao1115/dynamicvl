import json
import os
from os.path import join, abspath
from typing import Optional, List

from dvl.vqa import subset_mapping

import PIL
import numpy as np
from PIL import Image
from PIL.Image import Resampling
from torch.utils.data import Dataset


prompt_libs = dict(
        multi_choice="""Your task is to examine a chronological series of remote sensing images over a consistent geographical area, recorded across different years.
Your objective is to unearth the changes and continuities to answer the question below. Review the provided choices and select all that align with your findings.

<image>

Question:
{prompt}

Choices:
{options_str}

Submit ONLY the capital letter(s) for your confirmed findings.
- If a single finding is confirmed, answer its letter, such as "B".
- If multiple findings are confirmed, list their letters separated by a comma and a space, such as "C, D, E"
No further commentary or annotations are permitted in your answer.""",

        single_choice="""Your task is to examine a chronological series of remote sensing images over a consistent geographical area, recorded across different years.
Your objective is to unearth the changes and continuities to answer the question below. Review the provided choices and select the one that align with your finding.

<image>

Question:
{prompt}

Choices:
{options_str}

Submit ONLY the capital letter for your confirmed finding. Directly answer its letter, such as "B".
No further commentary or annotations are permitted in your answer.""",

        eco_assessment="""Your task is to interpret the provided image and answer the question below.
From the potential choices offered, please identify the one that most accurately answer the question.

Question:
{prompt}

Choices:
{options_str}

Please provide ONLY the capital letter corresponding to your answer choice, such as "C", with no additional explanations.""",

        basic_change_report_qa="""Your task is to examine a chronological series of remote sensing images over a consistent geographical area, recorded across different years.
Your objective is to identify and report basic land cover changes.

<image>

Generate a concise summary (1-3 sentences) that adheres to the following:
1. Time Period: Begin by stating the full observation period (e.g., "Over the YYYY–YYYY period," or "Over the X-year period from YYYY to YYYY,").
2. Key Transitions: Detail the primary land cover transitions observed, focusing on changes between major land cover types, such as vegetated areas, non-vegetated surfaces, and built-up areas/buildings.
3. Quantification: Report these transitions using percentage ranges (e.g., X-Y%) or approximate percentages (e.g., ~X%).""",

        change_speed_report_qa="""Your task is to examine a chronological series of remote sensing images over a consistent geographical area, recorded across different years.
Your objective is to determine and report the building area changes between consecutive image dates, including expansions, shrinkages, and areas with no change.

<image>

Generate a report string that adheres to the following:
1. Calculate the building area change rates between consecutive years (including expansion, shrinkage, and no change periods)
2. Output the results in the following format (choose one of the following for each time period):
"The changes were as follows: X% expansion from [Start Year] to [End Year]" OR "Y% shrinkage from [Start Year] to [End Year]" OR "no significant change from [Start Year] to [End Year]"

Example: "The expansion rates were as follows: 1.09% from 2006 to 2009, 1.1% from 2009 to 2011, 2.34% from 2011 to 2014.""",

        regional_caption="""Your task is to examine a chronological series of remote sensing images over a consistent geographical area, recorded across different years.
Your objective is to densely describe all change events within the red-boxed area over the time period covered by these images.

<image>

Generate a description string that adheres to the following:
1. Segment descriptions by distinct year ranges (e.g., YYYY-YYYY).
2. For each period, detail the initial land cover, specific changes observed (construction, clearing, feature additions/removals), and explicitly note any stability.
3. Use concise, factual language focusing on visible features like vegetation, buildings, infrastructure, and land states, including spatial references where relevant.""",

        dense_temporal_caption="""Your task is to examine a chronological series of remote sensing images over a consistent geographical area, recorded across different years.
Your objective is to describe the dynamics of this area over the time period covered by these images.

<image>

Generate a description string that adheres to the following:
1. Start with a general summary statement outlining the main trend over the entire period.
2. Detail specific land feature changes chronologically, using clear spatial references (e.g., top-left, center-right).
3. Conclude with an optional overall summary of the changes or final state.""",
)


class DynamicVLVQA(Dataset):
    def __init__(
            self,
            subset: str,
            data_dir: str,
            image_size: Optional[int] = None,
            messages_by_video: bool = False
    ):
        if subset in subset_mapping:
            subset = subset_mapping[subset]

        subset_json = abspath(join(data_dir, f"{subset}.json"))
        print("Reading data from {}".format(subset_json))
        with open(subset_json, "r") as f:
            data = json.load(f)
            data = [dict(id=f"{subset}_{data_idx}", **data_dict) for data_idx, data_dict in enumerate(data)]

        self.data = []
        miss_num = 0
        for doc in data:
            skip_flag = False
            if "image_list" in doc:
                for image_relpath in doc["image_list"]:
                    image_path = join(data_dir, image_relpath)
                    if not os.path.exists(image_path):
                        skip_flag = True
                        break
            elif "image_path" in doc:
                image_path = join(data_dir, doc["image_path"])
                if not os.path.exists(image_path):
                    skip_flag = True

            if skip_flag:
                miss_num += 1
                continue
            self.data.append(doc)

        if miss_num > 0:
            print(f"[WARN] Skipped {miss_num} samples in {subset_json} since their images are not found!")

        self.data_dir = data_dir
        self.subset = subset
        self.image_size = image_size
        self.messages_by_video = messages_by_video

    def filter_finish_ids(self, finish_ids: List[str]):
        self.data = [data_dict for data_dict in self.data if data_dict["id"] not in finish_ids]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        doc = self.data[idx]

        if self.subset in [
            "basic_change_choice_qa",
            "change_speed_choice_qa",
            "basic_change_report_qa",
            "change_speed_report_qa",
            "dense_temporal_caption",
            "regional_caption",
        ]:
            if self.subset in ["basic_change_choice_qa", "change_speed_choice_qa"]:
                if doc["task_type"] == "Single choice":
                    prompt_key = "single_choice"
                elif doc["task_type"] == "Multiple choice":
                    prompt_key = "multi_choice"
                else:
                    raise ValueError(f"Unknown task_type: {doc['task_type']}")

                prompt_text = prompt_libs[prompt_key].format(prompt=doc["prompts"], options_str=doc["options_str"])
            else:
                prompt_text = prompt_libs[self.subset]

            prompt_splits = prompt_text.split("<image>")
            assert len(prompt_splits) == 2, len(prompt_splits)

            messages = [{"role": "user", "content": [{"type": "text", "text": prompt_splits[0]}]}]

            # relative path to absolute
            doc["image_list"] = [join(self.data_dir, image_path) for image_path in doc["image_list"]]
            images = doc["image_list"]

            if self.messages_by_video:
                video_prefix = (
                        "These are the images captured in " +
                        ", ".join([image_year[:4] for image_year in doc["time_stamps"]]) +
                        ".\n"
                )
                messages[0]["content"].append({"type": "text", "text": video_prefix})
                messages[0]["content"].append({"type": "video", "video": images})

            else:
                for image_path, image_year in zip(images, doc["time_stamps"]):
                    image_year = image_year[:4]
                    messages[0]["content"].append({"type": "text", "text": f"\nyear {image_year}:"})
                    messages[0]["content"].append({"type": "image", "image": image_path})

            messages[0]["content"].append({"type": "text", "text": prompt_splits[1]})

        elif self.subset in ["eco_assessment"]:
            prompt_text = prompt_libs[self.subset].format(prompt=doc["prompts"], options_str=doc["options_str"])
            doc["image_path"] = join(self.data_dir, doc["image_path"])

            images = [doc["image_path"]]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": doc["image_path"]},
                        {"type": "text", "text": prompt_text},
                    ]
                }
            ]

        else:
            raise ValueError('Unknown subset {}'.format(self.subset))

        images = [Image.open(image_path).convert("RGB") for image_path in images]
        if self.image_size is not None:
            for img_idx, image in enumerate(images):
                images[img_idx] = image.resize((self.image_size, self.image_size), resample=Resampling.BICUBIC)

        # return:
        # - "images": List[PIL.Image]
        # - "messages": List[Dict]
        # - "metadata": Dict
        #   - "task_type": str "Single choice" or "Multiple choice"
        #   - "prompts": str
        #   - "options_str": str
        #   - "image_list": List[str]
        #   - "time_stamps": List[str]
        return dict(
            images=images,
            messages=messages,
            metadata=doc
        )



SEMANTIC_NAME_TO_LABEL = {
    'vegetation': 1,
    'non vegetated surface': 2,
    'water': 3,
    'building': 4,
    'playground': 5
}


class DynamicVLReferSeg(Dataset):
    prompt_template = """Your task is to examine 2 remote sensing images over a consistent geographical area.
Your objective is to segment the specific region indicated by the text query below.

Pre-image: <image>
Post-image: <image>

{prompt}"""

    def __init__(
            self,
            data_dir: str,
            image_size: Optional[int] = None,
    ):
        subset_json = abspath(join(data_dir, "change_referring_seg_qa.json"))
        print("Reading data from {}".format(subset_json))
        with open(subset_json, "r") as f:
            self.data = json.load(f)
            self.data = [dict(id=f"change_referring_seg_qa_{data_idx}", **data_dict) for data_idx, data_dict in enumerate(self.data)]

        self.data_dir = data_dir
        self.image_size = image_size

    def __len__(self):
        return len(self.data)

    def filter_finish_ids(self, finish_ids: List[str]):
        self.data = [data_dict for data_dict in self.data if data_dict["id"] not in finish_ids]

    def __getitem__(self, idx):
        doc = self.data[idx]

        image_years = [time_stamp[:4] for time_stamp in doc["time_stamps"]]
        t1_image_idx = image_years.index(doc["cd_years"][0])
        t2_image_idx = image_years.index(doc["cd_years"][1])

        t1_image_path = doc["image_list"][t1_image_idx]
        t2_image_path = doc["image_list"][t2_image_idx]

        t1_image_path = join(self.data_dir, t1_image_path)
        t2_image_path = join(self.data_dir, t2_image_path)

        t1_image = np.array(PIL.Image.open(t1_image_path).convert("RGB"), dtype=np.uint8)
        t2_image = np.array(PIL.Image.open(t2_image_path).convert("RGB"), dtype=np.uint8)

        gt_mask_path = join(self.data_dir, doc['ground_truth'])
        gt_mask = np.array(Image.open(gt_mask_path).convert("L"), dtype=np.uint8)
        gt_mask = (gt_mask > 0).astype(np.uint8)

        src_cls, tgt_cls = gt_mask_path.split("_")[-1][:-4].split("2")
        src_idx, tgt_idx = SEMANTIC_NAME_TO_LABEL[src_cls], SEMANTIC_NAME_TO_LABEL[tgt_cls]

        ref_text = self.prompt_template.format(prompt=doc["prompts"])
        if not ref_text.endswith("."):
            ref_text = ref_text + "."
        ref_text += " Please output segmentation mask."

        return dict(
            t1_image=t1_image,
            t2_image=t2_image,
            gt_mask=gt_mask,
            messages=[{"role": "user", "content": ref_text}, {"role": "assistant", "content": "[SEG]."}],
            cd_info=dict(
                src_cls=src_cls,
                tgt_cls=tgt_cls,
                src_idx=src_idx,
                tgt_idx=tgt_idx,
            ),
            metadata=doc,
        )
