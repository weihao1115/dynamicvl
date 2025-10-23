import argparse
import asyncio
import copy
import json
import logging
import os
import random
from os.path import dirname

import dvl.vqa

from typing import Dict, List
from openai import AsyncAzureOpenAI
from tqdm.asyncio import tqdm_asyncio


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


prompt_template = {
    "BCA-Report": """You are an advanced intelligent chatbot specialized in evaluating remote sensing image change detection results for basic land cover changes. 
Your primary task is to meticulously compare the predicted change description with the ground truth description and assess their accuracy. 

To accomplish this, you will evaluate the results across three key dimensions:

1. Land Cover Type Identification Accuracy:
Evaluate how accurately the predicted result identifies the main types of land cover changes, including:
* Correctly identifying the initial and final land cover types involved in the change
* Correctly describing the direction of transitions between different land cover types
* Capturing all major land cover transition patterns mentioned in the ground truth

2. Time Period Accuracy: 
Assess how accurately the predicted result captures the correct time period mentioned in the ground truth result. This measures whether the overall start year and end year are correctly identified.

3. Change Quantification Accuracy:
Assess how well the predicted result quantifies the magnitude of changes, including:
* Accuracy of the reported percentage changes
* Alignment between predicted quantitative changes and ground truth transition percentages
* Completeness of quantitative information compared to ground truth

Please assign a score for each of these three dimensions, using an integer from 0 to 5, where 5 indicates perfect performance and 0 signifies poor performance. Accompany your assessments with brief explanations to clarify your scoring rationale.

<predicted_result>
{predicted_result}
</predicted_result>

<ground_truth_result>
{ground_truth_result}
</ground_truth_result>

### OUTPUT FORMAT(EXAMPLE)
<land_cover_type_identification_accuracy>4</land_cover_type_identification_accuracy>
<land_cover_type_identification_explanation>
The prediction correctly identifies the main land cover types (vegetation, non-vegetated surfaces, built-up areas) and their transition patterns. It accurately captures the conversion from vegetation to built-up areas and non-vegetated surfaces. However, it misses some minor transition patterns mentioned in the ground truth.
</land_cover_type_identification_explanation>

<time_period_accuracy>5</time_period_accuracy>
<time_period_accuracy_explanation>
The predicted time period (2006–2021) exactly matches the ground truth time period (15-year period from 2006 to 2021).
</time_period_accuracy_explanation>

<change_quantification_accuracy>3</change_quantification_accuracy>
<change_quantification_explanation>
The prediction provides reasonable estimates of overall changes (~20% built-up increase, ~20% vegetation decrease) that roughly align with the ground truth's transition percentages. However, it focuses on net changes rather than specific transition flows and lacks the detailed quantification found in the ground truth.
</change_quantification_explanation>""",

    "CSE-Report": """You are an advanced intelligent chatbot specialized in evaluating remote sensing image change detection results.
Your primary task is to meticulously compare the predicted change detection results with the ground truth results and assess their accuracy. To accomplish this, you will evaluate the results across three key dimensions:

1. Change Rate Precision: 
Evaluate how accurately the percentage value in the predicted result matches the actual change rate described in the ground truth result. This measures whether the predicted rate is correct (without considering other aspects).

2. Time Period Accuracy: 
Assess how accurately the predicted result captures the correct time period mentioned in the ground truth result. This measures whether the start year and end year are correctly identified.

3. Change Pattern Accuracy:
Evaluate how accurately the predicted result describes the specific pattern and nature of changes, including:
* Correctly identifying the type of change (e.g., expansion, decrease, conversion)
* Accurately describing the spatial distribution or pattern of change
* Properly capturing the change dynamics mentioned in the ground truth

Please assign a score for each of these three dimensions, using an integer from 0 to 5, where 5 indicates perfect performance and 0 signifies poor performance. Accompany your assessments with brief explanations to clarify your scoring rationale.

<predicted_result>
{predicted_result}
</predicted_result>

<ground_truth_result>
{ground_truth_result}
</ground_truth_result>

### OUTPUT FORMAT(EXAMPLE)
<change_rate_precision>4</change_rate_precision>
<change_rate_precision_explanation>
The predicted expansion rate of 1.15% is very close to the ground truth rate of 1.09%, with only a small deviation of 0.06%.
</change_rate_precision_explanation>

<time_period_accuracy>5</time_period_accuracy>
<time_period_accuracy_explanation>
The predicted time period (2006 to 2009) exactly matches the ground truth time period.
</time_period_accuracy_explanation>

<change_pattern_accuracy>3</change_pattern_accuracy>
<change_pattern_accuracy_explanation>
The prediction correctly identifies urban expansion as the main change type. However, it provides limited information about the spatial distribution of changes and doesn't fully capture the detailed change dynamics described in the ground truth.
</change_pattern_accuracy_explanation>""",

    "DTC": """You are an advanced intelligent chatbot specialized in evaluating dense temporal captioning for remote sensing image time series.

Your primary task is to meticulously compare the predicted dense temporal caption with the ground truth caption and assess their accuracy. To accomplish this, you will evaluate the captions across three key dimensions:

1. Temporal Coverage: 
Evaluate how well the predicted caption captures all significant time points and periods of change throughout the entire temporal range. This includes identifying key temporal milestones, maintaining a logical sequence of events, providing appropriate temporal context, and capturing the complete temporal narrative without significant gaps.

2. Spatial Accuracy: 
Assess how accurately and comprehensively the predicted caption describes the spatial aspects of changes. This includes correctly identifying all regions where significant changes occurred, accurately describing the spatial relationships between different areas, using precise spatial referencing, and ensuring comprehensive coverage of all spatially relevant changes in the image.

3. Process Fidelity:
Evaluate how accurately and completely the predicted caption describes the nature and processes of change. This includes correctly identifying initial and final land cover/use states, describing intermediate stages of development, capturing the complexity of multiple change processes, and accurately describing the specific features that changed.

Please assign a score for each of these three dimensions, using an integer from 0 to 5, where 5 indicates perfect performance and 0 signifies poor performance. Accompany your assessments with brief explanations to clarify your scoring rationale.

<predicted_caption>
{predicted_result}
</predicted_caption>

<ground_truth_caption>
{ground_truth_result}
</ground_truth_caption>

### OUTPUT FORMAT(EXAMPLE)
<temporal_coverage>4</temporal_coverage>
<temporal_coverage_explanation>
The predicted caption successfully identifies the overall 15-year timeframe and captures key temporal milestones including the development initiation, 2011 halt, 2016 resumption, and 2018-2021 development phase. The chronological order is logical and comprehensive, though it could provide more specific details about the timing of the initial development phase. The caption maintains a clear temporal progression that allows readers to understand the sequence of changes over time.
</temporal_coverage_explanation>

<spatial_accuracy>5</spatial_accuracy>
<spatial_accuracy_explanation>
The caption precisely identifies all significant spatial regions where changes occurred including the top-right, left region, bottom-center, and bottom-right areas with accurate spatial referencing. The spatial coverage is comprehensive, capturing all relevant areas of change mentioned in the ground truth caption. The spatial descriptions effectively communicate the distribution of changes across the image and establish clear relative positions of different development activities.
</spatial_accuracy_explanation>

<process_fidelity>4</process_fidelity>
<process_fidelity_explanation>
The caption accurately describes the transformation from natural and agricultural lands to residential areas with impervious surfaces, buildings, and roads. It captures the phased development process and specifically mentions the office building construction and new water body. The description provides good detail about the conversion processes and infrastructure development. It could provide slightly more detail about the specific characteristics of the residential development in the left region to achieve perfect fidelity.
</process_fidelity_explanation>""",

    "RCC": """You are an advanced intelligent chatbot specialized in evaluating regional captions for remote sensing images.

Your primary task is to meticulously compare the predicted regional caption with the ground truth regional caption and assess their accuracy. To accomplish this, you will evaluate the captions across four key dimensions:

1. Temporal Coverage: 
Evaluate how well the predicted caption captures all significant time points and periods of change within the specified region. This includes identifying key temporal milestones, maintaining a logical sequence of events, and capturing the complete temporal narrative without significant gaps.

2. Spatial Accuracy: 
Assess how accurately the predicted caption describes the spatial aspects of changes within the region. This includes correctly identifying sub-areas where changes occurred and accurately describing the spatial relationships between different features.

3. Process Fidelity:
Evaluate how accurately the predicted caption describes the nature and processes of change. This includes correctly identifying initial and final land cover/use states, describing intermediate stages of development, and accurately describing the specific features that changed.

4. Region Containment:
Assess whether the caption strictly focuses on changes within the specified region box only, without including irrelevant information about areas outside the designated region. This measures the caption's precision in adhering to the spatial boundaries defined by the region box.

Please assign a score for each of these four dimensions, using an integer from 0 to 5, where 5 indicates perfect performance and 0 signifies poor performance. Accompany your assessments with brief explanations to clarify your scoring rationale.

<predicted_caption>
{predicted_result}
</predicted_caption>

<ground_truth_caption>
{ground_truth_result}
</ground_truth_caption>

### OUTPUT FORMAT(EXAMPLE)
<temporal_coverage>4</temporal_coverage>
<temporal_coverage_explanation>
The predicted caption effectively identifies the three key time periods (2005-2019, 2019-2020, 2020-2024) when changes occurred in the region. The temporal progression is logical and provides a clear understanding of the sequence of development events, though it could provide slightly more detail about specific milestones within the 2020-2024 period.
</temporal_coverage_explanation>

<spatial_accuracy>5</spatial_accuracy>
<spatial_accuracy_explanation>
The caption accurately identifies the specific sub-areas within the region where changes occurred, correctly referencing the "top left farmland" and "bottom left farmland." The spatial relationships are clearly described, providing an excellent understanding of where changes took place within the specified region.
</spatial_accuracy_explanation>

<process_fidelity>4</process_fidelity>
<process_fidelity_explanation>
The caption accurately describes the conversion of farmland to residential areas and the development process including ground hardening. It effectively captures the progression from agricultural use to development and construction. The description of the final state of the bottom left farmland could be more detailed to achieve perfect fidelity.
</process_fidelity_explanation>

<region_containment>5</region_containment>
<region_containment_explanation>
The caption focuses exclusively on the changes occurring within the specified region box, with no references to areas outside the designated boundaries. All described features and changes are properly contained within the region of interest, demonstrating perfect adherence to the spatial constraints.
</region_containment_explanation>"""
}


async def process_item(client, data_item, model_name) -> Dict:
    data_item = copy.deepcopy(data_item)

    retry_num, break_flag = 0, False
    response = None
    while not break_flag:
        try:
            completion_outputs = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model_name,
                    messages=data_item["request"]["messages"],
                    max_tokens=data_item["request"]["max_tokens"],
                ),
                timeout=300
            )
            try:
                response = completion_outputs.choices[0].message.content
            except AttributeError:
                response = None

            break_flag = True

        except (asyncio.TimeoutError, ConnectionError) as e:
            logger.error(f"Network exception occurred: {e}")

        except Exception as e:
            if (
                "rate limit" in str(e).lower()
                or "too many requests" in str(e).lower()
                or "server error" in str(e).lower()
                or "connection error" in str(e).lower()
            ):
                logger.error(f"Sever error occurred for: {e}")
            else:
                logger.error(f"Other error occurred for: {e}. skip it")
                break_flag = True

        if not break_flag:
            retry_num = retry_num + 1
            if retry_num > 10:
                logger.info(f"Max Retry reached for, skip it")
                break_flag = True
            else:
                wait_time = min(2 ** retry_num, 10)
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)

    return dict(
        id=data_item["custom_id"],
        response=response,
    )


async def process_with_semaphore(client, data_item, model_name, semaphore):
    async with semaphore:
        await asyncio.sleep(random.uniform(0.1, 0.5))
        return await process_item(client, data_item, model_name)


async def main(
    batched_requests: List[Dict],
    response_save_path: str,
    model_name: str = "gpt-4.1-mini",
    n_thread: int = 128,
    overwrite: bool = False
):
    n_thread = min(n_thread, len(batched_requests))

    client = AsyncAzureOpenAI(
        azure_endpoint=os.environ.get("AZURE_OPENAI_BASE"),
        api_key=os.environ.get("AZURE_OPENAI_KEY"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    )
    try:
        finished_ids = []
        if not overwrite and os.path.exists(response_save_path):
            with open(response_save_path, "r") as f:
                for line in f.readlines():
                    data_dict = json.loads(line.strip())
                    if data_dict.get("response"):
                        finished_ids.append(data_dict["id"])
        batched_requests = [data_dict for data_dict in batched_requests if data_dict["custom_id"] not in finished_ids]
        if len(batched_requests) == 0:
            print("All tasks already completed. Skipping.")
            return

        tasks = []
        semaphore = asyncio.Semaphore(n_thread)
        for data_item in batched_requests:
            tasks.append(
                process_with_semaphore(
                    client=client,
                    data_item=data_item,
                    model_name=model_name,
                    semaphore=semaphore
                )
            )

        with open(response_save_path, "a" if len(finished_ids) > 0 else "w") as f:
            for task in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
                parsed_response = await task
                f.write(json.dumps(parsed_response) + "\n")
                f.flush()

    finally:
        if client and hasattr(client, 'close'):
            await client.close()

        for task in asyncio.all_tasks(asyncio.get_running_loop()):
            if task is not asyncio.current_task() and not task.done():
                task.cancel()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_model_id", type=str, default="gpt-4.1-mini")
    parser.add_argument("--eval_model_id", type=str, required=True)
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument('--result_dir', type=str, default=f"{dirname(dvl.__file__)}/../../results/vqa")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--n_thread", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.subset not in prompt_template:
        raise ValueError(f"Invalid subset: {args.subset}. Valid options: {list(prompt_template.keys())}")

    model = args.eval_model_id.replace("/", "--")
    print(f"Processing {model}")

    result_json_path = f"{args.result_dir}/{model}/{args.subset}.json"
    if os.path.exists(result_json_path):
        with open(result_json_path) as f:
            result_data = json.load(f)

        batched_requests = []
        for doc_idx, doc in enumerate(result_data):
            pred = doc["response"]
            gt = doc["metadata"]["ground_truth"]

            request_dict = dict(
                custom_id=doc["id"] if "id" in doc else doc_idx,
                request=dict(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt_template[args.subset].format(predicted_result=pred, ground_truth_result=gt),
                        }
                    ],
                    max_tokens=args.max_tokens
                )
            )
            batched_requests.append(request_dict)

        response_save_path = result_json_path.replace(".json", f".{args.gpt_model_id}.jsonl")
        asyncio.run(
            main(
                batched_requests=batched_requests,
                response_save_path=response_save_path,
                model_name=args.gpt_model_id,
                n_thread=args.n_thread,
                overwrite=args.overwrite,
            )
        )
    else:
        print(f"Result file not found: {result_json_path}")
