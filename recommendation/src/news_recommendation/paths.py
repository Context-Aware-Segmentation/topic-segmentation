"""Root of the recommendation part; configured paths are relative to it (absolute paths pass through)."""
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
