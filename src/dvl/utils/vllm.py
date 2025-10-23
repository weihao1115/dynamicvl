from typing import Dict

from transformers import GenerationConfig
from vllm import SamplingParams


def get_sampling_params(model_id: str, override_sampling_params: Dict = None, follow_hf: bool = False):
    sampling_params = SamplingParams()
    try:
        if follow_hf:
            default_generation_config = GenerationConfig.from_pretrained(model_id).to_dict()
        else:
            default_generation_config = GenerationConfig.from_pretrained(model_id).to_diff_dict()
        for key in default_generation_config:
            if hasattr(sampling_params, key):
                setattr(sampling_params, key, default_generation_config[key])
        sampling_params.update_from_generation_config(default_generation_config)
    except OSError:
        print(f"No HF generation config file found for {model_id}, using default VLLM sampling params.")
        pass

    if override_sampling_params is not None:
        for k, v in override_sampling_params.items():
            if v is not None:
                if hasattr(sampling_params, k):
                    setattr(sampling_params, k, v)
                else:
                    print(f"{k} not found in VLLM sampling params!")

    if sampling_params.temperature > 0:
        sampling_params.temperature = max(sampling_params.temperature, 0.01)
    return sampling_params
