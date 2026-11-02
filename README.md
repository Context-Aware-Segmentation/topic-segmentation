# Structure-Aware News Topic Segmentation with Silver-Label Augmentation

Code for Yeyang Liu's master's thesis at the University of Zurich: news topic segmentation from editorial subheadings, clinical transfer, and applications to information extraction and recommendation.

Read the [online documentation](https://mt-docs.pages.dev/) or [build the three documentation sites locally](docs/README.md).

## Research questions and parts

| # | Research question | Part |
|---|---|---|
| RQ1 | How can editorial structure be used to train effective news topic segmentation models, and how well do these models transfer to clinical documents? | [Topic segmentation](topic_segmentation/README.md) |
| RQ2 | How does input segmentation affect information extraction for knowledge graph construction in news and clinical documents? | [Knowledge graph](knowledge_graph/README.md) |
| RQ3 | How does representing news articles through summaries of predicted topic segments affect recommendation ranking, compared with abstract and whole-article representations? | [Recommendation](recommendation/README.md) |

Each part has its own `src/`, `scripts/`, and README. Segmentation and recommendation use `configs/`; knowledge-graph settings live beside the model adapters.

## Setup

Install [`uv`](https://docs.astral.sh/uv/), enter the part you want to use, and activate its locked environment:

```bash
cd topic_segmentation  # or knowledge_graph / recommendation
source scripts/activate.sh
```

Follow the part's README for experiment commands and its setup guide for data and model assets: [segmentation](topic_segmentation/SETUP.md), [knowledge graph](knowledge_graph/SETUP.md), [recommendation](recommendation/SETUP.md). Recommendation summary generation uses a separate environment from ranking.

## Data and model availability

Datasets, model weights, and run outputs are not in git.

- **MIMIC-IV-Note** requires credentialed PhysioNet access. No MIMIC text, MIMIC-derived prediction, or MIMIC-trained model is distributed with this repository.
- **MIND, C4, and DWIE** must be obtained from their original sources.

## Attribution

The segmentation code builds on Yu et al. (2023); see its [citations](topic_segmentation/README.md#citations). Other sources are listed under [knowledge-graph third-party code](knowledge_graph/README.md#third-party-code) and [recommendation third-party code](recommendation/README.md#third-party-code).

See [third-party notices](THIRD_PARTY_NOTICES.md) for the applicable licenses and components whose upstream repositories do not state a license.
