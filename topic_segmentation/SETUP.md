# Environment Setup & Run Commands

From the repository root, enter `topic_segmentation/` and run the commands below:

```bash
cd topic_segmentation
```

## Installation

```bash
# uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create and activate the locked environment, repeat in every new shell
source scripts/activate.sh

# Punkt sentence splitter for IDEA-seg and MIND; activate.sh points NLTK to this folder
python -m nltk.downloader -d data/resources/nltk punkt
```

News, MIND, and clinical training use `cache/tmp/` unless `TMPDIR` is set. C4 preprocessing keeps its temporary Arrow files in `cache/c4/`.

## Data and weights

IDEA-seg, the MIND article bodies, and the trained thesis models are not distributed with the repository. The public models are downloaded below; C4 is downloaded during preparation; MIMIC-IV-Note requires credentialed PhysioNet access.

### Models

```bash
hf download answerdotai/ModernBERT-base --local-dir checkpoints/modernbert_base
hf download google-bert/bert-base-uncased --local-dir checkpoints/bert_base
hf download FacebookAI/roberta-base --local-dir checkpoints/roberta_base
hf download allenai/longformer-base-4096 --local-dir checkpoints/longformer_base

# Chonkie baselines
hf download minishlab/potion-base-32M --local-dir checkpoints/baselines/chonkie/semantic-potion-base-32m
hf download mirth/chonky_modernbert_base_1 --local-dir checkpoints/baselines/chonkie/neural-modernbert-base
```

### Data

#### IDEA-seg

The source articles are available in the [IDEA dataset](https://github.com/Informfully/Datasets/tree/main/IDEA). IDEA-seg adds the segmentation labels and splits used in this thesis.

- `data/news/`: train/val/test files, split into paragraphs.
- `data/news_sent/`: the same articles, split into sentences.
- `data/news/raw/`: saved pages, article list, and source dataset.

The pipeline uses the supplied splits in `data/news/` and `data/news_sent/`. To rebuild IDEA-seg separately from saved pages:

```bash
python scripts/prepare_data.py news
```

The rebuilt dataset and splits go in `runs/news_construction/<time>/`.

#### MIND

Download the original dataset from the [MIND website](https://msnews.github.io/). The download does not include article bodies; these must be supplied separately (see the [official data description](https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md#newstsv)).

Put MIND articles with full text in `data/mind/full/news.csv`. Columns:

```text
news_id,category,subvert,title,abstract,url,entity,ab_entity,body
```

Standalone preparation writes predicted boundaries to `data/mind/full/` and training splits to `data/mind/prepared/`. The pipeline writes both under its MIND run's `data/` folder. Subset sizes are 2k, 5k, 10k, and 25k articles.

#### C4 `realnewslike`

Preparation downloads `realnewslike` from [allenai/c4 on Hugging Face](https://huggingface.co/datasets/allenai/c4). Tokenized data with paragraph-boundary labels goes in `data/c4/tokenized_modernbert/`.

#### MIMIC-IV-Note 2.2

Download [MIMIC-IV-Note 2.2 from PhysioNet](https://physionet.org/content/mimic-iv-note/2.2/) after completing the required training, obtaining credentialed access, and signing the data use agreement.

- Raw notes: `data/mimic/raw/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/`.
- Prepared evaluation and training data: `data/mimic/`.

### Check

With IDEA-seg and the four backbones in place, check versions, data, and tokenizers offline:

```bash
python scripts/checkenv.py
```

---

## Run Commands

These are independent examples using configured inputs. Use `run_ts_pipeline.sh` to chain newly trained checkpoints automatically.

Stage scripts resolve relative paths from `topic_segmentation/` and accept `--dry-run` (print commands) and `--help`.

```bash
# ── 1. Data ───────────────────────────────────────────────────────────────────
# MIND comes after stage 3: its silver labels need the news teacher (see below)
python scripts/prepare_data.py c4 mimic

# ── 2. Boundary pre-training on C4 ────────────────────────────────────────────
# Smoke test on 100k examples
python scripts/run_boundary_pretraining.py --smoke-test --gpus 0

# Full run
python scripts/run_boundary_pretraining.py --gpus 0 1

# ── 3. News fine-tuning ───────────────────────────────────────────────────────
# Experiment groups: backbones, boundary_pretraining, granularity, objectives, clinical_initialization
python scripts/run_news_finetuning.py --experiment backbones --gpu 0
python scripts/run_news_finetuning.py --experiment objectives --gpu 0

# One thesis run
python scripts/run_news_finetuning.py --run modernbert_c4_1024_cssl --gpu 0

# Any setting: --backbone bert|roberta|longformer|modernbert|modernbert_c4, --length 512|1024,
# --objective ts|ts_da|cssl|cssl_da|tssp|tssp_da|full_noda|full, --data news|news_sent
python scripts/run_news_finetuning.py --backbone roberta --length 512 --objective ts_da --gpu 0
python scripts/run_news_finetuning.py --backbone modernbert_c4 --length 1024 --objective full --data news_sent --gpu 0

# ── MIND data with the configured teacher; --teacher-model selects a new one ──
python scripts/prepare_data.py mind --gpu 0

# ── 4. MIND adaptation ────────────────────────────────────────────────────────
# Experiment groups: scale, final, objective_matrix, matched_steps, clinical_initialization
python scripts/run_mind_adaptation.py --experiment final --gpu 0

# One thesis run (reruns the adaptation it starts from)
python scripts/run_mind_adaptation.py --run mind_25k_full_news_ts --gpu 0

# One thesis run, starting from the saved adaptation
python scripts/run_mind_adaptation.py --run mind_25k_full_news_ts --recorded-adaptation --gpu 0

# Any setting: --size 2k|5k|10k|25k, --objective ts|ts_da|full, --matched-steps, --news-objective ts|ts_da|full
python scripts/run_mind_adaptation.py --size 10k --objective full --matched-steps --news-objective ts --gpu 0

# ── 5. Long-document inference ────────────────────────────────────────────────
# Experiment groups: strides, stage_matrix, matched_steps, baselines
python scripts/run_long_document_inference.py --experiment strides --gpu 0

# Any model and window setting
python scripts/run_long_document_inference.py --model runs/mind_adaptation/mind_25k_full_news_ts \
    --window 1024 --stride 256 512 768 1024 --no-protect --gpu 0
python scripts/run_long_document_inference.py --model runs/mind_adaptation/mind_25k_full_news_ts \
    --window 1024 --stride 512 --protect --gpu 0

# Chonkie baselines
python scripts/run_long_document_inference.py --baseline semantic --gpu 0
python scripts/run_long_document_inference.py --baseline neural --gpu 0

# ── 6. Clinical transfer ──────────────────────────────────────────────────────
# Train the news initializations first
python scripts/run_news_finetuning.py --experiment clinical_initialization --gpu 0
python scripts/run_mind_adaptation.py --experiment clinical_initialization --gpu 0

# Zero-shot: --models news_pretrain news_ft chonkie_semantic chonkie_neural, --levels block subheading
python scripts/run_clinical_transfer.py zero-shot --gpus 0
python scripts/run_clinical_transfer.py zero-shot --models news_ft --levels block --gpus 0

# Clinical fine-tuning on growing training sets: --initialization base news_pretrain news_ft
python scripts/run_clinical_transfer.py scaling --gpus 0 1
python scripts/run_clinical_transfer.py scaling --initialization base --gpus 0

# ── All stages in thesis order ────────────────────────────────────────────────
# Set GPU IDs and experiment groups in the script
bash scripts/run_ts_pipeline.sh --dry-run
bash scripts/run_ts_pipeline.sh
```

## Using reproduced checkpoints

`run_ts_pipeline.sh` passes the new C4, news, and MIND checkpoints to their downstream stages. It uses one timestamp across `runs/<stage>/reproductions/<time>/` and trains each shared model once.

MIND sentences, teacher predictions, subset files, and prepared splits go in the current MIND run's `data/` folder. Clinical predictions and fine-tuning results go in `zero_shot/` and `scaling/` under the current clinical run.

For individual stage commands, the options below select reproduced checkpoints. Example output directories:

```bash
c4="$PWD/runs/boundary_pretraining/reproductions/my_run"
news="$PWD/runs/news_finetuning/reproductions/my_run"
mind="$PWD/runs/mind_adaptation/reproductions/my_run"
clinical="$PWD/runs/clinical_transfer/reproductions/my_run"
```

### After boundary pre-training

The C4 run directory contains the final model and tokenizer. Use it for news fine-tuning and MIND adaptation with `--c4-model`:

```bash
python scripts/run_boundary_pretraining.py --gpus 0 1 --output-dir "$c4"
python scripts/run_news_finetuning.py --experiment backbones boundary_pretraining granularity objectives clinical_initialization \
    --c4-model "$c4" --output-dir "$news" --gpu 0
```

`--experiment` accepts several groups and deduplicates their shared runs.

### After news fine-tuning

Use `modernbert_c4_1024_ts` as the MIND teacher and write its silver labels to a new directory:

```bash
python scripts/prepare_data.py mind --teacher-model "$news/modernbert_c4_1024_ts" \
    --output-dir "$mind/data" --gpu 0
```

`--output-dir` redirects all MIND preparation outputs; the raw articles still come from the configured dataset.

### MIND adaptation

Train adaptations from the new C4 model and silver labels, then fine-tune them on news:

```bash
python scripts/run_mind_adaptation.py --experiment scale final objective_matrix matched_steps clinical_initialization \
    --c4-model "$c4" --prepared-root "$mind/data/prepared" --output-dir "$mind" --gpu 0
```

Each adaptation runs before the news models that depend on it. To run only news fine-tuning, use `--from-adaptation DIR` to reuse adaptations from that directory, or `--recorded-adaptation` to reuse the saved thesis adaptations.

### News inference

Use the reproduced directories for all model references in an inference experiment:

```bash
python scripts/run_long_document_inference.py --experiment stage_matrix \
    --news-models-dir "$news" --mind-models-dir "$mind" --gpu 0
```

### Clinical transfer

Use the same new checkpoints for zero-shot inference and clinical fine-tuning:

```bash
python scripts/run_clinical_transfer.py zero-shot --gpus 0 \
    --news-ft "$news/modernbert_c4_1024_full_sentences" \
    --news-pretrain "$mind/mind_25k_cssl_news_ts" --output-dir "$clinical"
python scripts/run_clinical_transfer.py scaling --gpus 0 1 \
    --news-ft "$news/modernbert_c4_1024_full_sentences" \
    --news-pretrain "$mind/mind_25k_cssl_news_ts" --output-dir "$clinical"
```

Use a new clinical output directory when changing checkpoints: scaling skips completed runs in an existing directory.

### Direct inference

Pass a checkpoint directly and select the device:

```bash
python -m topic_segmentation.inference --model "$mind/mind_25k_full_news_ts" --input data/sample_input.jsonl --device cuda:0
```
