<h1 align="center">DynamicVL: Benchmarking Multimodal Large Language Models for Dynamic City Understanding</h1>
<p align="center">
  <a href="https://arxiv.org/abs/2505.21076">
    <img src="https://img.shields.io/badge/ArXiv-2505.21076-b31b1b.svg?logo=arXiv" alt="ArXiv">
  </a>
  <a href="https://huggingface.co/datasets/weihao1115/dvl_suite">
    <img src="https://img.shields.io/badge/HuggingFace-Dataset-yellow.svg?logo=huggingface&logoColor=white" alt="Hugging Face Dataset">
  </a>
</p>
<h5 align="center">
  <em>Weihao Xuan, Junjue Wang, Heli Qi, Zihang Chen, Zhuo Zheng, Yanfei Zhong, Junshi Xia, Naoto Yokoya</em>
</h5>

## About
DynamicVL is a comprehensive framework for analyzing long-term urban dynamics through remote sensing imagery. This repository ships the DVL-Suite dataset, task-specific benchmarks, and evaluation scripts that cover both closed-form vision-language tasks and pixel-level change detection.

## News
- **2025/08** &nbsp; DynamicVL was accepted to NeurIPS 2025! We will add encoder-decoder-based semantic change detection implementations to this repo. Stay tuned!



### Environment Setup
```bash
# Create the conda environment
conda create -n dvl python=3.10 -y
conda activate dvl

# Install the package
(dvl): pip install -e .

# Optional: manually install PyTorch if the vLLM dependency conflicts with your environment
# Note: Downgrade cu128 if it conflicts with your CUDA drivers.
(dvl): pip install -U torch torchvision xformers --index-url https://download.pytorch.org/whl/cu128

# Optional: fix "version `GLIBCXX_3.4.32' not found" errors
(dvl): conda install -c conda-forge gcc=13 gxx=13 -y
(dvl): export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

## Data Setup
Download the DVL-Suite dataset and unzip the training and test archives:
```bash
mkdir data && cd data
unzip train.zip
unzip test.zip
```

Expected directory layout:
```
data/
├── train/                          # DVL-Instruct (Training Set)
│   ├── images/{city}/{region}/{image_id_timestamp}.tif
│   ├── cd_sem_masks/
│   ├── cd_refer_seg_masks/
│   ├── regional_caption/
│   ├── metadata.json
│   ├── basic_change_choice_qa.json
│   ├── basic_change_report_qa.json
│   ├── change_speed_choice_qa.json
│   ├── change_speed_report_qa.json
│   ├── change_referring_seg_qa.json
│   ├── eco_assessment.json
│   ├── dense_temporal_caption.json
│   └── regional_caption.json
└── test/                           # DVL-Bench (Test Set)
    └── [same structure as train/]
```

## Usage

### Vision-Language Tasks

#### Load Data
```python
from dvl.vqa.dataset import DynamicVLVQA

dataset = DynamicVLVQA(subset="BCA-QA", data_dir="data/train")
for item in dataset:
    # images: List[PIL.Image] across time
    # messages: multi-turn Q&A dicts
    # metadata: contains id, task_type, prompts, options_str, image_list, time_stamps
    print(item)
```

#### Evaluate Open-Source Models (vLLM)
```bash
(dvl): python -m dvl.vqa.run_vllm \
    --model_id Qwen/Qwen2.5-VL-3B-Instruct \
    --subset BCA-QA
```
**Available subsets:**
- `BCA-QA` - Basic Change Analysis (QA)
- `CSE-QA` - Change Speed Estimation (QA)
- `BCA-Report` - Basic Change Analysis (Report)
- `CSE-Report` - Change Speed Estimation (Report)
- `DTC` - Dense Temporal Caption
- `RCC` - Regional Change Caption
- `EA` - Environmental Assessment

> **Note:** Set `--batch_size 1` for `llava-hf/llava-onevision-qwen2-7b-ov-hf` to avoid GPU OOM.

**Output:** `results/vqa/Qwen--Qwen2.5-VL-3B-Instruct/` stores `.jsonl` predictions and `.json` summaries.

#### Evaluate Commercial Models (Azure OpenAI)
```bash
export AZURE_OPENAI_BASE="{your-azure-endpoint}"
export AZURE_OPENAI_KEY="{your-api-key}"
export AZURE_OPENAI_API_VERSION="{your-api-version}"

(dvl): python -m dvl.vqa.run_azure_openai \
    --model_id gpt-4o \
    --subset BCA-QA
```
**Output:** `results/vqa/gpt-4o/` stores task-specific `.jsonl` predictions and `.json` metrics.

#### GPT-Based Evaluation for Reports and Captions
```bash
export AZURE_OPENAI_BASE="{your-azure-endpoint}"
export AZURE_OPENAI_KEY="{your-api-key}"
export AZURE_OPENAI_API_VERSION="{your-api-version}"

(dvl): python -m dvl.vqa.pretty_print.gpt_eval \
    --gpt_model_id gpt-4.1-mini \
    --eval_model_id "Qwen/Qwen2.5-VL-3B-Instruct" \
    --subset DTC
```
**Supported subsets:**
- `BCA-Report`
- `CSE-Report`
- `DTC`
- `RCC`

**Output:** `results/vqa/Qwen--Qwen2.5-VL-3B-Instruct/` includes GPT-scored `.jsonl` files (for example `DTC.gpt-4.1-mini.jsonl`).

#### Aggregate Metrics
```bash
# Multi-choice QA tasks (BCA-QA, CSE-QA, EA)
(dvl): python -m dvl.vqa.pretty_print.acc_table

# Open-ended generation tasks (Reports & Captions)
(dvl): python -m dvl.vqa.pretty_print.gen_table --gpt_model_id gpt-4.1-mini
```
Tabulated metrics are printed to console and saved in `results/vqa/`.

### Referring Change Detection

#### Load Data
```python
from dvl.vqa.dataset import DynamicVLReferSeg

dataset = DynamicVLReferSeg(data_dir="data/train")
for item in dataset:
    # t1_image, t2_image: np.ndarray of shape (1024, 1024, 3)
    # gt_mask: binary change mask
    # messages: instruction-response history
    # cd_info: source/target land-cover classes and indices
    # metadata: contains the unique evaluation id
    print(item)
```

#### Evaluate Predictions
Organize predicted masks using `item["metadata"]["id"]` as the filename stem:
```
{your-pred-dir}/
├── change_referring_seg_qa_0.png
├── change_referring_seg_qa_1.png
└── ...
```

Run the evaluation utilities:
```bash
# LISA-style binary IoU metrics
(dvl): python -m dvl.vqa.pretty_print.referseg_iou --pred_dir "{your-pred-dir}"

# MambaCD-style semantic change detection metrics
(dvl): python -m dvl.vqa.pretty_print.referseg_cd --pred_dir "{your-pred-dir}"
```
Scores are printed to console and stored alongside the submitted prediction masks.

## Citation
If you find DynamicVL useful, please cite:
```bibtex
@article{xuan2025dynamicvl,
  title={DynamicVL: Benchmarking Multimodal Large Language Models for Dynamic City Understanding},
  author={Xuan, Weihao and Wang, Junjue and Qi, Heli and Chen, Zihang and Zheng, Zhuo and Zhong, Yanfei and Xia, Junshi and Yokoya, Naoto},
  journal={arXiv preprint arXiv:2505.21076},
  year={2025}
}
```

## License
DynamicVL is released under the **Apache-2.0 License**.


## Acknowledgements
DynamicVL builds on NAIP aerial imagery and the open-source multimodal community. We appreciate all contributors who benchmarked cutting-edge MLLMs on our dataset and shared feedback during the public release.
