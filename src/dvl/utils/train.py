import os

import torch


def get_unique_job_id() -> str:
    job_id = ""
    # slurm case
    if "SLURM_JOB_ID" in os.environ:
        job_id = os.environ["SLURM_JOB_ID"]
    if "SLURM_ARRAY_JOB_ID" in os.environ:
        job_id = f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}"
    # sge case
    if "JOB_ID" in os.environ:
        job_id = os.environ["JOB_ID"]
    if "SGE_TASK_ID" in os.environ and os.environ["SGE_TASK_ID"] != "undefined":
        job_id = f"{os.environ['JOB_ID']}_{os.environ['SGE_TASK_ID']}"
    # PJM case
    if "PJM_JOBID" in os.environ:
        job_id = os.environ["PJM_JOBID"]
    if "PBS_JOBID" in os.environ:
        job_id = os.environ["PBS_JOBID"]
    return job_id


def is_global_zero() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True

