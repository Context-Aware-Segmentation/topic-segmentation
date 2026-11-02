# Code structure

`news/` contains the MIND-50 and DWIE experiments. `clinical/` contains the Synthetic-100 experiments. They share file I/O, metric counts, checkpoint selection, and prediction interfaces.

## News

The news pipeline follows these dependencies:

```text
Raw articles → segmentation → partitions
                              ├─ entity recognition and linking → NER/EL scores
                              └─ DocRED inputs → training → relation predictions → RE scores
```

`data.py` prepares documents, partitions, and segments for extraction. `inputs.py` tokenizes each document and aligns its entities once, then selects sentences for each input condition and records the entity mappings. Window lengths use the models' tokenizer and entity markers. `entity_linking.py` runs spaCy or ReLiK. `policies.py` combines their spans; `score.py` evaluates the resulting spans and links.

`docred.py` prepares tokens, entity pairs, and offsets. Training adds relation targets; DREEAM also adds sentence-evidence targets. Batches use named fields and carry only the targets needed by the caller. `encoding.py` handles chunk and overlapping-window encoding, layer aggregation, and entity pooling. `docre.py` loads ATLOP/DREEAM and runs inference from DocRED records. Both models return relation predictions and, when labels are supplied, their losses. `train.py` handles optimization and validation. `relations.py` maps predictions back to document entities and scores them. The relation models remain in `atlop.py` and `dreeam.py`.

DREEAM evidence uses sentences containing both endpoints, or the nearest pair of endpoint sentences. Equal-distance pairs are ordered by subject sentence ID, then object sentence ID. Training normalizes evidence targets over sentences reached by the pair's context attention; pairs with no reachable evidence contribute only to the relation loss.

Training initializes the encoder from `roberta-base`. Inference builds the architecture from its configuration and loads the encoder and classifier weights from the selected relation checkpoint.

`scripts/news.sh` connects these stages; see [SETUP](../SETUP.md#news-pipeline-paths) for pipeline paths and [relation-extraction commands](../SETUP.md#3-news-relation-extraction) for training and checkpoint selection.

## Clinical

`data.py` reads the prepared notes, splits the IDs in `manifest.json`, and builds the input segments of each condition. `schema.py` defines the Clinical-10 entity types, relation signatures, model prompts, and annotation schema. `ner.py` runs the released NER models; `finetune.py`, `medcat.py`, and `phenobert.py` adapt their models and score the test notes. `gliner_relex.py` evaluates joint GLiNER-Relex extraction and extraction over reference entities, `glirel.py` trains and evaluates GLiREL, and `hpo.py` compares the HPO linkers. `scoring.py` holds the NER and RE metrics.

Released and fine-tuned models share GLiNER-Relex inference settings, span decoding, and reference alignment. Each PhenoBERT backend owns its model weights and classifier cache, so released and fine-tuned models can run in the same process.

Generation and annotation are separate commands in `synthetic.py`, `annotation.py`, and `hpo_annotation.py`; `agents.py` runs their `codex exec` jobs. Their validators return the errors that are sent back to the agent.

## Prediction contracts

- NER and news NERL receive a document and its segments. Each segment carries `text`, `begin`, `end`, and `segment_id`.
- Returned character offsets refer to the full document and use exclusive ends: `text[begin:end]` selects the mention.
- NER retains model-specific labels and concept IDs. News NERL retains Wikidata IDs or Wikipedia titles.
- DocRED relation inputs contain tokenized sentences and entity clusters. Mappings connect their local entity indices to document entities.
- GLiREL receives tokens and entity spans with inclusive token ends. Its returned endpoint spans use exclusive ends, as consumed by the alignment code.
- Gold entity alignment is used for scoring predicted mentions. Saved relations with unmatched endpoints carry `"mapped": false` and remain false positives when rescored.
- An input condition with no candidate entity pairs returns no relations. Its gold relations still count as false negatives.

Model adapters load their weights once and implement `predict()`. Runners own file I/O and scoring.

## Python use

The model adapters load their weights once and expose `predict(document, segments)`: the clinical NER models in `clinical.ner` and `SpacyLinker` and `ReLiK` in `news.entity_linking`. Each segment carries `text`, `begin`, `end`, and `segment_id`; returned character offsets refer to the full document, with exclusive ends.

```python
from knowledge_graph.clinical.ner import GLiNER

model = GLiNER()  # uses CUDA
document = {"id": "example", "text": "The patient has a headache."}
segments = [{"text": document["text"], "begin": 0, "end": len(document["text"]), "segment_id": 0}]
entities = model.predict(document, segments)
```

`news.docre.DocREPredictor` takes DocRED records (`title`, `sents`, `vertexSet`) and returns `(title, head, relation, tail)` tuples: `predict()` reads inputs of up to 1,024 subwords, `predict_with_windows()` whole documents of up to 2,048 subwords through overlapping encoder windows.

For a 2,048-subword single-pass checkpoint, construct the predictor with `single_pass=True` and use `predict(max_length=2048)` for whole documents. `predict()` retains the 1,024-subword cap for other inputs; both calls encode each input once.

## Paths and outputs

`paths.py` defines asset and output roots. See [SETUP](../SETUP.md#run-options) for path overrides and run directories.

Training selects checkpoints using validation scores; test scoring uses the selected weights. Clinical adaptation also uses early stopping.
