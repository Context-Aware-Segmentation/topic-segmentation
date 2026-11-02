# Environment Setup & Run Commands

From the repository root, enter `recommendation/` and run the commands below:

```bash
cd recommendation
```

## Installation

```bash
# Install uv once
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create and activate the locked ranking environment; repeat in each new shell
source scripts/activate.sh
```

Segmentation, summary generation, and ranking use separate environments because their Transformers versions differ. `replay` and ranking use only the first environment; regeneration needs all three.

| Step | Environment |
|---|---|
| Data preparation, training | `source scripts/activate.sh`: Python 3.11, PyTorch 2.6.0 (CUDA 12.4), Transformers 4.57.6 |
| Segmentation | The [topic-segmentation environment](../topic_segmentation/SETUP.md#installation) |
| Summary generation | `UV_PROJECT_ENVIRONMENT="${MT_UV_ENV_HOME:-$HOME/.local/share/uv/mt}/news-summary-generation" uv sync --project generation --locked`: Transformers 5.9.0 and lm-format-enforcer |

The pipeline finds the segmentation and generation interpreters in the same environment folder; `--segmentation-python` and `--generation-python` override them.

## Data and weights

Download the labeled MIND-large train and dev sets from the [MIND website](https://msnews.github.io/). The original download does not include full article bodies or the saved segmentation and summaries used here; supply those thesis assets separately.

| Path | Contents |
|---|---|
| `data/mind/large/{train,dev}/` | Labeled MIND-large `news.tsv` and `behaviors.tsv` |
| `checkpoints/bert_base/` | BERT-base-uncased |
| `checkpoints/summarizers/Qwen3-4B-Instruct-2507/` | Summary model (regeneration only) |
| `data/mind/large/summaries/summary_full_catalog/segmented.jsonl` | Saved full-text catalog with predicted segments |
| `data/mind/large/summaries/section_summary_40_50_full/outputs.jsonl` | Saved segment summaries |
| `data/mind/large/summaries/article_summary_250_350/` | Saved article summaries |
| `data/mind/large/filtered_by_day/` | Saved split and packed inputs, including the segment selection |

[configs/data_preparation.json](configs/data_preparation.json) names the replay sources. Regeneration also needs the segmenter, the full-text catalog, and the Punkt model named in [configs/segmentation.json](configs/segmentation.json), all in `../topic_segmentation/`. Configured paths are relative to `recommendation/`.

Download the public model weights:

```bash
hf download google-bert/bert-base-uncased --local-dir checkpoints/bert_base
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir checkpoints/summarizers/Qwen3-4B-Instruct-2507
```

The Qwen model is needed only for regeneration.

## Data preparation

`news_recommendation.preparation` pools the labeled MIND-large train and dev sets. It keeps news whose URL matches exactly one full-text body with at least 600 BERT tokens and at most ten predicted segments, and that has an abstract. Impressions keep these news and need at least five history clicks, five candidates, and both click labels. The result has 19,065 articles and 571,267 impressions: November 9–13, 2019 for training (355,522), November 14 for validation (106,270), and November 15 for test (109,475).

Abstracts and article summaries are cut at the end of the text. Segment summaries keep their order and are dropped whole when they do not fit.

## Run commands

### Replay saved inputs

```bash
python -m news_recommendation.pipeline replay --output runs/replay
```

Add `--train` to train and evaluate the three representations after preparation.

### Train one representation

```bash
python scripts/run_training.py title_section_summary_512 \
    --data runs/replay/data --gpus 0 1 2 3 --output-dir runs/reproduction
```

Omit the representation name to run all three. Training settings are in [configs/plm_nr.json](configs/plm_nr.json). The model uses the first eight BERT layers and trains only layers with indices 6 and 7. Training starts from the configured initialization; the saved checkpoints are for evaluation.

### Regenerate segmentation and summaries

```bash
python -m news_recommendation.pipeline regenerate --output runs/regenerated --summary-gpus 0 1
```

Add `--train` to train and evaluate the regenerated inputs.

Regeneration uses two GPUs; the article-summary long stage places half the model layers on each. It segments the full-text catalog with the news segmenter (1,024-token windows, stride 512) and summarizes the retained articles with Qwen3-4B-Instruct-2507 under greedy decoding. Segment summaries are asked for 15–40 words and accepted at 10–50; article summaries are asked for 250–350 words and accepted whenever they end.

Regenerated text is not guaranteed to match the saved summaries token for token because batches differ. `replay` reproduces the thesis inputs exactly.
