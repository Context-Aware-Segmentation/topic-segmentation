# Third-party code

The licenses below apply to the identified components, not to the repository as a whole. Paper citations are in each module's README.

## Transformers

[Transformers](https://github.com/huggingface/transformers) is distributed under [Apache-2.0](licenses/Apache-2.0.txt).

- `topic_segmentation/src/topic_segmentation/training.py` adapts its token-classification example through Yu et al. (2023), with segmentation objectives, data preparation, and experiment settings. The source retains the HuggingFace Team's 2020 copyright notice.
- `knowledge_graph/src/knowledge_graph/news/relik_loading.py` adapts `PoolerEndLogits` from Transformers 4.41 to load the pinned ReLiK reader. It retains the original file's copyright notices.

## DREEAM

The relation and evidence model in `knowledge_graph/src/knowledge_graph/news/dreeam.py` is adapted from [DREEAM](https://github.com/YoumiMa/dreeam).

Copyright (c) 2023 YoumiMa. The full [MIT license](licenses/DREEAM-MIT.txt) is included. Local changes connect the model to shared encoding, input preparation, and evaluation.

## PhenoBERT and fastNLP

`knowledge_graph/src/knowledge_graph/clinical/vendor/phenobert/` contains an adapted subset of [PhenoBERT](https://github.com/EclipseCN/PhenoBERT). Copyright (c) 2020 NeoFengyh. Its [MIT license](knowledge_graph/src/knowledge_graph/clinical/vendor/phenobert/LICENSE) is retained.

The nested `fastNLP/` directory comes from [fastNLP](https://github.com/fastnlp/fastNLP/tree/v0.4.10), under [Apache-2.0](licenses/Apache-2.0.txt), separately from PhenoBERT's MIT license. That historical release declares version 0.4.5. Local changes retain the components needed for PhenoBERT inference and model loading, adjust imports, and remove unused utilities and documentation. Modified source files carry a notice.

## ATLOP

`knowledge_graph/src/knowledge_graph/news/atlop.py` is adapted from [ATLOP](https://github.com/wzhouad/ATLOP). Its feature preparation and batching also inform `news/docred.py`.

The upstream repository does not state a license. Attribution is retained; this repository does not assign a license or grant redistribution permission for that upstream code.

## PLM4NewsRec

`recommendation/src/news_recommendation/model.py` is adapted from [PLM4NewsRec](https://github.com/wuch15/PLM4NewsRec) by Chuhan Wu and colleagues. The adaptation retains its BERT title encoder, attention pooling, and checkpoint parameter names.

The upstream repository does not state a license. Attribution is retained; this repository does not assign a license or grant redistribution permission for that upstream code.
