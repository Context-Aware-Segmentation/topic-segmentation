"""Check offline that the segmentation environment, the news data, and the local backbones are ready.

    python scripts/checkenv.py
"""
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = ("bert_base", "roberta_base", "longformer_base", "modernbert_base")


def main():
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")  # before Hugging Face is imported
    import datasets
    import torch
    import transformers
    from transformers import AutoTokenizer

    from topic_segmentation import prepare_inputs

    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPUs), "
          f"transformers {transformers.__version__}, datasets {datasets.__version__}")
    with tempfile.TemporaryDirectory() as cache:
        news = prepare_inputs.load_splits(ROOT / "data/news", cache_dir=cache)
        assert {split: len(rows) for split, rows in news.items()} == {"train": 709, "validation": 89, "test": 89}
        del news  # Close Arrow mappings before the temporary cache is removed on NFS.
    for name in BACKBONES:
        AutoTokenizer.from_pretrained(ROOT / "checkpoints" / name)
    print("environment check passed")


if __name__ == "__main__":
    main()
