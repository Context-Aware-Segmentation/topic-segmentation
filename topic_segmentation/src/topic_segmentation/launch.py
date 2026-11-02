import os
import sys
from pathlib import Path

from topic_segmentation.paths import REPOSITORY


TRAINING_DEFAULTS = [
    "--do_train", "True",
    "--do_eval", "True",
    "--do_predict", "True",
    "--load_best_model_at_end", "True",
    "--save_total_limit", "2",
    "--metric_for_best_model", "overall_f1",
    "--eval_accumulation_steps", "1000",
    "--preprocessing_num_workers", "5",
    "--gradient_checkpointing", "False",
    "--ts_loss_weight", "1.0",
    "--cl_positive_k", "1",
    "--cl_negative_k", "3",
]


def command(model, dataset, data_dir, profile, objective, output, cache, max_length=None, max_steps=None):
    arguments = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        "--module",
        "topic_segmentation.training",
        "--model_name_or_path", str(model),
        "--dataset_name", dataset,
        "--data_dir", str(data_dir),
        "--dataset_cache_dir", str(cache),
        "--seed", str(profile["seed"]),
        "--max_seq_length", str(max_length or profile["max_sequence_length"]),
        "--learning_rate", str(profile["learning_rate"]),
        "--per_device_train_batch_size", str(profile["per_device_batch_size"]),
        "--gradient_accumulation_steps", str(profile["gradient_accumulation_steps"]),
        "--per_device_eval_batch_size", str(profile["per_device_batch_size"]),
        "--eval_strategy", "steps",
        "--eval_cnt", str(profile["evaluation_points"]),
        "--do_da_ts", str(objective["do_da_ts"]),
        "--do_tssp", str(objective["do_tssp"]),
        "--cl_loss_weight", str(objective["cl_loss_weight"]),
        "--cl_temp", str(objective.get("cl_temp", 0.1)),
        "--tssp_loss_weight", str(objective["tssp_loss_weight"]),
        "--output_dir", str(output),
    ]
    arguments += TRAINING_DEFAULTS
    if max_steps:
        return arguments + ["--max_steps", str(max_steps)]
    return arguments + ["--num_train_epochs", str(profile["epochs"])]


def environment(gpu):
    """Configure one visible GPU and caches, preserve TMPDIR, and disable torch.compile."""
    variables = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        HF_MODULES_CACHE=str(REPOSITORY / "cache/hf_modules"),
        HF_DATASETS_CACHE=str(REPOSITORY / "cache/hf_datasets"),
        TMPDIR=os.environ.get("TMPDIR", str(REPOSITORY / "cache/tmp")),
        TORCHDYNAMO_DISABLE="1",
        OMP_NUM_THREADS="8",
    )
    Path(variables["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    return variables
