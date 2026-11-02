"""KG assets and output locations; KG_CORPUS and KG_RUNS select the clinical corpus and run root."""

import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
DATA = REPOSITORY / "data"
RUNS = Path(os.environ.get("KG_RUNS", REPOSITORY / "runs")).expanduser().resolve()
SEGMENTATION = REPOSITORY.parent / "topic_segmentation"  # MIMIC notes and MIND segmentations
MIMIC_DATA = SEGMENTATION / "data/mimic"
SYNTHETIC100 = Path(os.environ.get("KG_CORPUS", DATA / "synthetic100")).expanduser().resolve()
