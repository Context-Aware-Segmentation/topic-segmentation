# Environment Setup & Run Commands

From the repository root:

```bash
cd knowledge_graph
```

## Installation

The locked environment uses Python 3.11 and PyTorch 2.6 with CUDA 12.4. spaCy and MedCAT run on CPU; the neural training and relation-extraction commands use CUDA. ATLOP and DREEAM use FP32 weights with BF16 autocast for training, validation, and inference.

```bash
# Install uv once
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Install dependencies and activate; repeat in each new shell
source scripts/activate.sh

# NLTK resources for PhenoBERT
python -m nltk.downloader -d data/resources/nltk stopwords wordnet averaged_perceptron_tagger_eng
```

The environment is stored in `~/.local/share/uv/mt/knowledge-graph/` by default. Set `MT_UV_ENV_HOME` before activation to choose its parent directory.

Check the installation:

```bash
python -c "import torch, spacy, knowledge_graph; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('spaCy:', spacy.__version__)"
```

The news segmentation stage also uses the sibling package's environment. Follow its [installation instructions](../topic_segmentation/SETUP.md#installation).

## Data and weights

### MIND-50

The original dataset is available from the [MIND website](https://msnews.github.io/). These experiments use a prepared set of 50 articles with full text and entity annotations:

- `data/mind50/mind_nerl_gold_qid.json`: articles, reference mentions, and Wikidata IDs.
- `../topic_segmentation/data/mind/full/news_seg_predictions.jsonl`: sentence records used to build segmentation inputs.

Use the thesis copies of these files. They are additional assets beyond the original MIND download.

### DWIE

The [DWIE repository](https://github.com/klimzaporojets/DWIE) provides the original corpus. The loader here reads the [DFKI-SLT Parquet mirror](https://huggingface.co/datasets/DFKI-SLT/DWIE/blob/refs%2Fconvert%2Fparquet/Task_1/train/0000.parquet):

```bash
mkdir -p data/dwie
curl -fL 'https://huggingface.co/datasets/DFKI-SLT/DWIE/resolve/refs%2Fconvert%2Fparquet/Task_1/train/0000.parquet' \
    -o data/dwie/train.parquet
```

This file includes both training and test documents; the code reads their split tags. Place the saved gold title-to-QID cache at `data/dwie/dwie_title_qid_cache.json`.

News scoring starts from the saved title-to-QID cache `runs/news/final/prediction_title_qids.json` and looks up missing titles online. Index comparisons use the candidate QIDs that an index provides and look up titles only for the remaining predictions. The Wikipedia-title score excludes gold spans annotated only with `wikidata:Q…`.

### Synthetic-100

Place the prepared corpus under `data/synthetic100/`:

```text
data/synthetic100/
├── manifest.json
├── sources/                         # note text, blocks, and sections
├── reviews/                         # reference entities and relations
└── hpo_adjudication/review_final/    # reference HPO links
```

The code derives the shared 80/10/10 split from the 100 note IDs in `manifest.json` with seed 42. Set `KG_CORPUS` to use a different corpus directory for generation, annotation, training, and scoring. See [Regenerate Synthetic-100](#regenerate-synthetic-100) for its construction steps.

MIMIC-derived assets are kept outside Git.

### HPO and FastHPOCR

Place the thesis ontology and index together:

- `data/resources/hpo/hp.obo`: HPO release `2026-06-23`.
- `data/resources/hpo/hp.index`: the saved FastHPOCR index.

The [HPO project](https://github.com/obophenotype/human-phenotype-ontology) provides ontology releases. [FastHPOCR](https://github.com/tudorgroza/fast_hpo_cr#indexing-hpo) documents how to build an index for an ontology file.

### News models

Place the trained weights at:

```text
../topic_segmentation/runs/mind_adaptation/mind_25k_full_news_ts/
runs/news/fixed/docre_models/atlop/best.pt
runs/news/fixed/docre_models/dreeam/best.pt
```

ReLiK downloads its weights on first use. Relation-model training downloads `roberta-base`. Inference uses the supplied model weights but still needs the RoBERTa configuration and tokenizer, either cached locally or downloaded. spaCy's language models are installed with the environment; spaCy-entity-linker downloads its database on first use. To download that database ahead of time:

```bash
python -m spacy_entity_linker download_knowledge_base
```

### MedCAT

Download the MedCAT v1 SNOMED 2025 model pack through the [MedCAT model download instructions](https://github.com/CogStack/MedCAT#available-models), using your UMLS credentials. The experiment expects:

```text
checkpoints/medcat/v1_Snomed2025_MIMIC_IV.zip
```

Keep the archive zipped; MedCAT loads the model pack directly.

### PhenoBERT

Download the data and pretrained files linked in the [PhenoBERT repository](https://github.com/EclipseCN/PhenoBERT#pretrained-embeddings-and-models). Arrange them as follows:

```text
data/phenobert/
├── hpo.json
├── stopwords.txt
└── NUM.txt

checkpoints/phenobert/
├── embeddings/
│   ├── biobert_v1.1_pubmed/
│   └── fasttext_pubmed.bin
└── models/
    ├── HPOModel_H/                  # model_layer1.pkl and model_l1_*.pkl
    └── bert_model_max_triple.pkl
```

The PhenoBERT loader uses the NLTK resources installed above and downloads Stanza's English MIMIC/i2b2 model on first use.

### Other clinical models

These models download on first use:

- GLiNER-Relex: `knowledgator/gliner-relex-large-v1.0`.
- GLiREL: `jackboyla/glirel-large-v0`.
- SapBERT: `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`.
- HunFlair2: Flair's `hunflair2` model.

## Run options

Run commands below from `knowledge_graph/` after activation. The [run commands](#run-commands) list the experiment sequence.

### Choose a GPU

Use `CUDA_VISIBLE_DEVICES` to select a GPU for one process:

```bash
CUDA_VISIBLE_DEVICES=1 python -m knowledge_graph.clinical.finetune --model gliner_relex
CUDA_VISIBLE_DEVICES=2 bash scripts/news.sh nerl runs/news/rerun
```

### News pipeline paths

`scripts/news.sh STAGE OUT` writes intermediate files and results to `OUT`. Use the same output directory across stages.

The stages are `segment`, `data`, `inputs`, `nerl`, `score`, `mentions`, `train`, and `relations`. `all` skips training; `reproduce` includes it and evaluates the newly trained models.

To use a different segmentation checkpoint:

```bash
SEG_MODEL=/path/to/segmentation-checkpoint bash scripts/news.sh segment runs/news/rerun
```

The script activates the KG and segmentation environments as needed. `KG_PYTHON` and `SEG_PYTHON` can specify alternative Python executables.

### Relation models

`relations` uses `OUT/docre_models/` when that directory exists, otherwise the saved thesis models; `MODELS=/path/to/docre_models` chooses another directory. To train one model after the `inputs` stage:

```bash
python -m knowledge_graph.news.train --implementation atlop \
    --inputs runs/news/rerun/docre --out-dir runs/news/rerun/docre_models
```

ATLOP and DREEAM train for ten epochs from RoBERTa and keep the checkpoint with the best development-set F1; `--epochs` (or `TRAIN_EPOCHS` for the Bash pipeline) changes the number of epochs. Other settings are at the top of [`news/train.py`](src/knowledge_graph/news/train.py).

Inputs up to 1,024 tokens use leading and trailing encoder chunks; a central chunk fills any gap. DREEAM normalizes evidence targets over the sentences with nonzero context attention of each positive entity pair. Relation inference evaluates every test condition in the input directory's `mappings.json`; whole documents are read through overlapping encoder windows (`encoder_windows`, 2,048-subword cap). The `document_level` result combines encoder-window predictions on long documents with 1,024-subword chunk predictions on short ones.

For the separately trained 2,048-subword single-pass checkpoints, use the Python entry point with `--single-pass-2048`:

```bash
python -m knowledge_graph.news.relations predict --implementation atlop \
    --inputs runs/news/rerun/docre \
    --model /path/to/singlepass2048/atlop/best.pt \
    --single-pass-2048 \
    --gold runs/news/rerun/dwie_test_gold.json \
    --out-dir runs/news/rerun_singlepass2048/atlop
```

Use `dreeam` with its own checkpoint, or `runs/news/rerun/predicted_docre` for predicted mentions, with a separate output directory for each run. The input directory must include the checkpoint's matching `rel2id.json`. These checkpoints contain 2,050 position vectors, initialized during training by tiling RoBERTa's 512 pretrained positions; inference restores the learned vectors directly.

Each input uses one encoder pass. Whole documents have a 2,048-subword cap and are saved as `document`; other conditions retain their 1,024-subword cap. Both caps include entity markers and special tokens. This mode does not produce the hybrid `document_level` result. The Bash pipeline and commands without this option retain the standard encoding.

`predictions.json` maps each input condition to its predicted relations; relations with an endpoint outside the gold mentions carry `"mapped": false` and count as false positives. To rescore a run:

```bash
python -m knowledge_graph.news.relations score \
    --gold runs/news/rerun/dwie_test_gold.json \
    --predictions runs/news/rerun/relations_predicted/atlop/predictions.json \
    --out runs/news/rerun/relations_predicted/atlop/results.json
```

### News input conditions and retrieval indexes

- No segment option: process each whole article.
- `--segments FILE`: use predicted topic segments from a segmentation JSONL file.
- `--partitions FILE --partition semantic`: use one DWIE partition; `document`, `equal`, and `random` are the others.
- `--index`: choose a ReLiK retrieval index.

For example, after preparing the MIND-50 segmentation predictions:

```bash
python -m knowledge_graph.news.entity_linking --backend relik \
    --input data/mind50/mind_nerl_gold_qid.json \
    --segments runs/news/rerun/mind50_predictions.jsonl \
    --index sapienzanlp/relik-retriever-e5-base-v2-aida-blink-wikipedia-2Mpopular-index \
    --out runs/news/rerun/mind50_relik_semantic_2mpopular.json
```

### Clinical runs

Clinical commands write under `$KG_RUNS/clinical/` (default `runs/clinical/`). To keep a reproduction apart from the saved thesis runs, set the run root before starting and keep it for every stage:

```bash
export KG_RUNS="${KG_RUNS:-$PWD/runs/reproduction}"
```

Training writes into the model's run folder, replacing earlier weights and scores there. `--evaluate-only` scores the saved best checkpoint again (`finetune`, `medcat`, `phenobert`). Relation predictions are stored per note as `predicted_relation_keys` and `gold_relation_keys`; `--notes test` selects the ten test notes and `--notes all` all 100.

Use `--model` to select another relation checkpoint. For GLiREL with predicted entities, `--ner-dir` selects the directory containing GLiNER-Relex `test_<condition>.json` files. See the [relation-extraction commands](#6-clinical-relation-extraction).

PhenoBERT fine-tuning builds phrase pairs from the reference HPO links; when normalized pairs conflict, it keeps the same-concept label over parent or unrelated labels.

## Run commands

### 1. News data and segmentation

Prepare MIND-50 and DWIE with the final news segmentation model:

```bash
bash scripts/news.sh segment runs/news/rerun
bash scripts/news.sh data runs/news/rerun
bash scripts/news.sh inputs runs/news/rerun
```

These stages create the segmentation predictions, DWIE partitions, DocRED-format relation inputs, and gold Wikidata links. DWIE compares whole documents, topic segments, equal partitions, and random partitions.

The default segmentation checkpoint is `../topic_segmentation/runs/mind_adaptation/mind_25k_full_news_ts`. Set `SEG_MODEL` to use a different checkpoint.

### 2. News entity recognition and linking

Run spaCy and ReLiK, then score exact entity spans and Wikidata links:

```bash
bash scripts/news.sh nerl runs/news/rerun
bash scripts/news.sh score runs/news/rerun
```

This also compares ReLiK retrieval indexes. The scoring stage starts from the saved title-to-QID cache at `runs/news/final/prediction_title_qids.json`.

### 3. News relation extraction

Build inputs from predicted mentions, then run ATLOP and DREEAM with both reference and predicted entities:

```bash
bash scripts/news.sh mentions runs/news/rerun
bash scripts/news.sh relations runs/news/rerun
```

Predictions and scores go to `relations_gold/` and `relations_predicted/` under the run directory. See [SETUP](#relation-models) to rescore saved predictions.

The default weights are `runs/news/fixed/docre_models/{atlop,dreeam}/best.pt`. To retrain and evaluate the new weights:

```bash
bash scripts/news.sh train runs/news/rerun
bash scripts/news.sh relations runs/news/rerun
```

Training writes each model's best checkpoint to `runs/news/rerun/docre_models/`.

With the saved weights and data in place, all news stages can be run together:

```bash
bash scripts/news.sh all runs/news/rerun # includes inference and scoring; skips training
```

Use `bash scripts/news.sh reproduce runs/news/rerun` to include training. Relation extraction uses the models trained in that output directory; `MODELS` selects an explicit checkpoint directory when running `relations` separately.

### 4. Clinical entity recognition

Synthetic-100 contains 100 synthetic discharge notes. Adaptation uses a shared 80/10/10 train/validation/test split with seed 42.

Commands read `data/synthetic100/` by default. For a regenerated corpus, keep `KG_CORPUS` and `KG_RUNS` set as shown in [SETUP](#regenerate-synthetic-100).

Each model is evaluated on six input conditions: `fixed128`, `fixed256`, `fixed512`, `sentence`, `block`, and `section`. Token budgets use the spaCy tokenizer; sentence, block, and section inputs are capped at 512 tokens.

Run the five released NER models:

```bash
export KG_RUNS="${KG_RUNS:-$PWD/runs/reproduction}"

for model in phenobert fasthpocr gliner_relex hunflair2 medcat; do
    python -m knowledge_graph.clinical.ner --model "$model"
done
```

Fine-tune GLiNER-Relex and HunFlair2, adapt MedCAT, and fine-tune PhenoBERT:

```bash
python -m knowledge_graph.clinical.finetune --model gliner_relex
python -m knowledge_graph.clinical.finetune --model hunflair2
python -m knowledge_graph.clinical.medcat
python -m knowledge_graph.clinical.phenobert
```

The released models run on all 100 notes; the comparisons use the ten test notes. Adapted models save their validation-selected weights, test predictions, and `test_results.json` under `${KG_RUNS}/clinical/<model>/`. NER uses exact character-span micro-F1 without requiring entity types to match.

Add `--evaluate-only` to any of these commands to score its saved best checkpoint again.

### 5. Clinical entity linking

Compare FastHPOCR, PhenoBERT, and SapBERT on reference and predicted symptom mentions. Run this after GLiNER-Relex, HunFlair2, and PhenoBERT adaptation:

```bash
python -m knowledge_graph.clinical.hpo prepare
python -m knowledge_graph.clinical.hpo fasthpocr
python -m knowledge_graph.clinical.hpo phenobert
python -m knowledge_graph.clinical.hpo sapbert
```

Results go to `${KG_RUNS}/clinical/hpo_linker_comparison/`.

### 6. Clinical relation extraction

Run released GLiNER-Relex with joint extraction and with reference entities, then evaluate the fine-tuned model with reference entities:

```bash
python -m knowledge_graph.clinical.gliner_relex joint --notes test \
    --out "${KG_RUNS}/clinical/gliner_relex_zero_shot_joint"
python -m knowledge_graph.clinical.gliner_relex reference-entities --notes test \
    --out "${KG_RUNS}/clinical/gliner_relex_zero_shot_gold"
python -m knowledge_graph.clinical.gliner_relex reference-entities --notes test \
    --model "${KG_RUNS}/clinical/gliner_relex/best" --out "${KG_RUNS}/clinical/gliner_relex_gold"
```

Fine-tune GLiREL, then compare its released and adapted weights with reference and predicted entities. Predicted entities come from the fine-tuned GLiNER-Relex model in stage 4.

```bash
python -m knowledge_graph.clinical.glirel finetune

for entities in gold predicted; do
    python -m knowledge_graph.clinical.glirel predict --notes test --entities "$entities" \
        --out-dir "${KG_RUNS}/clinical/glirel_zero_shot_$entities"
    python -m knowledge_graph.clinical.glirel predict --notes test --entities "$entities" \
        --model "${KG_RUNS}/clinical/glirel_adapted/best" \
        --out-dir "${KG_RUNS}/clinical/glirel_adapted_$entities"
done
```

## Regenerate Synthetic-100

The thesis experiments use the saved notes and reviewed annotations. Regeneration produces a new corpus.

Select its data and run directories before starting:

```bash
export KG_CORPUS="$PWD/data/synthetic100_regenerated"
export KG_RUNS="$PWD/runs/synthetic100_regenerated"
```

Keep both variables set when running the clinical experiments on this corpus.

### Generate notes

The case-card stage reads prepared MIMIC files from `../topic_segmentation/data/mimic/finetune_sets/`:

- `discharge_bhc_seed42_n101000/splits_valid1000_seed42/ids/train_100000.txt`.
- `discharge_bhc_seed42_n5000/discharge_random5000_by_id.json`.
- `discharge_bhc_seed42_n101000/discharge_random96000_by_id.json`.

See the [MIMIC setup](../topic_segmentation/SETUP.md#mimic-iv-note-22) for the source data. Put the Qwen teacher at `checkpoints/teachers/qwen3-32b-awq/`.

The case-card stage needs vLLM, which is separate from the locked KG environment. Set `VLLM_PYTHON` to a Python executable with vLLM installed:

```bash
VLLM_PYTHON=/path/to/vllm-env/bin/python
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH="$PWD/src:$PWD/../topic_segmentation/src" \
    "$VLLM_PYTHON" -m knowledge_graph.clinical.synthetic cards --tensor-parallel-size 2
```

The remaining generation stages use an authenticated Codex CLI:

```bash
python -m knowledge_graph.clinical.synthetic expand
python -m knowledge_graph.clinical.synthetic reformat
```

Outputs go to `$KG_CORPUS/generation/`. `cards` saves completed groups after each batch and skips them on rerun; `expand` and `reformat` keep every job whose response already passes validation.

Use `--model` and `--out` after the stage name to select the model and output directory. `--tensor-parallel-size` applies to `cards`; `--max-parallel` and `--reasoning-effort` apply to `expand` and `reformat`. Run a stage with `--help` for its options.

### Annotate and review

Prepare the generated notes in `$KG_CORPUS`, annotate entities and relations, then add HPO links. Rerunning `prepare` keeps existing annotations.

```bash
python -m knowledge_graph.clinical.annotation prepare
python -m knowledge_graph.clinical.annotation annotate
python -m knowledge_graph.clinical.annotation review
python -m knowledge_graph.clinical.annotation validate
python -m knowledge_graph.clinical.hpo_annotation adjudicate
python -m knowledge_graph.clinical.hpo_annotation review
```

Both HPO passes receive the full note text. The first pass also receives lexical candidates and local mention contexts.

The agent stages accept `--model`, `--reasoning-effort`, `--max-parallel`, `--max-attempts`, and `--note-ids` (a comma-separated subset). Outputs record the model, reasoning effort, and stage under `generation`. Reruns keep valid outputs, including human-reviewed annotations, and regenerate missing or invalid ones.
