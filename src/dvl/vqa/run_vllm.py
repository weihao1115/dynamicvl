import argparse
import json
import os
from dataclasses import asdict
from os.path import dirname, join
from torch.utils.data import Dataset
from tqdm import tqdm

from dvl.vqa import subset_mapping
from dvl.vqa.dataset import DynamicVLVQA
import dvl
import dvl.vqa

from vllm import EngineArgs, LLM

from dvl.utils import build_model_config, ModelConfig
from dvl.utils.vllm import get_sampling_params


def create_batch_inputs(ds: Dataset, model_config: ModelConfig, args):
    for i in range(0, len(ds), args.batch_size):
        batch_data = [ds[j] for j in range(i, min(i + args.batch_size, len(ds)))]
        batch_inputs = []
        batch_metadata = []

        for data_dict in batch_data:
            messages = data_dict["messages"]
            images = data_dict["images"]

            if "llava" in args.model_id.lower() and args.subset != "EA":
                prompt_text, multimodal_inputs = model_config.get_prompt_from_question(messages)
                inputs = {"prompt": prompt_text, **multimodal_inputs}
            else:
                prompt_text = model_config.get_prompt_from_question(messages)
                inputs = {
                    "prompt": prompt_text,
                    "multi_modal_data": {"image": images}
                }

            batch_inputs.append(inputs)
            batch_metadata.append(data_dict["metadata"])

        yield batch_inputs, batch_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--subset', type=str, required=True, choices=list(subset_mapping.keys()))
    parser.add_argument('--data_dir', type=str, default=f"{dirname(dvl.__file__)}/../../data/test")
    parser.add_argument('--result_dir', type=str, default=f"{dirname(dvl.__file__)}/../../results/vqa")
    parser.add_argument('--max_model_len', type=int, default=32768)
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument('--image_size', type=int, default=None)
    parser.add_argument('--tensor_parallel_size', type=int, default=None)
    parser.add_argument('--enforce_eager', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    reformat_model_id = args.model_id.replace('/', '--')
    if args.image_size is not None:
        reformat_model_id = reformat_model_id + f"_{args.image_size}"
    result_save_path = join(args.result_dir, reformat_model_id, f"{args.subset}.jsonl")
    os.makedirs(dirname(result_save_path), exist_ok=True)

    finish_ids = []
    if not args.overwrite and os.path.exists(result_save_path):
        with open(result_save_path) as f:
            for line in f.readlines():
                data_dict = json.loads(line.strip())
                finish_ids.append(data_dict["id"])

    if len(finish_ids) > 0:
        print(f"Resume from {len(finish_ids)} finished samples")
    else:
        print(f"Start from scratch")

    ds = DynamicVLVQA(
        subset=args.subset,
        data_dir=args.data_dir,
        image_size=args.image_size,
        messages_by_video="llava" in args.model_id.lower()
    )
    ds.filter_finish_ids(finish_ids)

    if len(ds) > 0:
        model_config = build_model_config(
            model_id=args.model_id,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens
        )
        engine_args = model_config.default_engine_args
        if "model" not in engine_args:
            engine_args["model"] = args.model_id

        if args.subset in [
            "BCA-QA",
            "CSE-QA",
            "BCA-Report",
            "CSE-Report",
            "DTC",
            "RCC",
        ]:
            if "llava" in args.model_id.lower():
                engine_args["limit_mm_per_prompt"] = dict(video=1)
            else:
                engine_args["limit_mm_per_prompt"] = dict(image=10)
        elif args.subset in ["EA"]:
            engine_args["limit_mm_per_prompt"] = dict(image=1)
        else:
            raise ValueError(f"Unknown subset {args.subset}")

        if args.tensor_parallel_size is not None:
            engine_args["tensor_parallel_size"] = args.tensor_parallel_size
        if args.enforce_eager:
            engine_args["enforce_eager"] = args.enforce_eager
        engine_args = asdict(EngineArgs(**engine_args))
        print(f"Engine arguments: {engine_args}")
        vlm_model = LLM(**engine_args)

        sampling_params = get_sampling_params(
            model_id=args.model_id,
            override_sampling_params=dict(max_tokens=args.max_tokens),
        )
        print(f"Sampling params: {sampling_params}")

        with open(result_save_path, "w" if len(finish_ids) == 0 else "a") as f:
            progress_bar = tqdm(total=len(ds))
            for batch_inputs, batch_metadata in create_batch_inputs(ds, model_config, args):
                outputs = vlm_model.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)

                for idx, output in enumerate(outputs):
                    generated_text = output.outputs[0].text
                    dump_dict = {
                        "id": batch_metadata[idx]["id"],
                        "response": generated_text,
                        "metadata": batch_metadata[idx],
                    }
                    f.write(json.dumps(dump_dict, ensure_ascii=False) + "\n")
                    f.flush()
                    progress_bar.update(1)

            progress_bar.close()

    with open(result_save_path) as f:
        finished_data = [json.loads(line.strip()) for line in f.readlines()]

    with open(result_save_path.replace(".jsonl", ".json"), "w") as f:
        json.dump(finished_data, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()
