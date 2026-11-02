#!/usr/bin/env bash
# Run the thesis topic-segmentation experiments in order.
#
#   bash scripts/run_ts_pipeline.sh
#   bash scripts/run_ts_pipeline.sh --dry-run    # print the commands only
#
# Edit the GPU settings and experiment groups below.
# Downstream stages use the checkpoints and MIND silver labels produced by this run.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
options=("$@")
gpu=0
gpus=(0 1)
run_id="$(date -u +%Y%m%dT%H%M%S%NZ)"
c4="$repo_root/runs/boundary_pretraining/reproductions/$run_id"
news="$repo_root/runs/news_finetuning/reproductions/$run_id"
mind="$repo_root/runs/mind_adaptation/reproductions/$run_id"
inference="$repo_root/runs/long_document_inference/reproductions/$run_id"
clinical="$repo_root/runs/clinical_transfer/reproductions/$run_id"
source "$repo_root/scripts/activate.sh"
cd "$repo_root"

run() {
    local script=$1
    shift
    python "scripts/$script" "$@" "${options[@]}"
}

# 1. Data; IDEA-seg splits are supplied in data/news/ and data/news_sent/.
run prepare_data.py c4
run prepare_data.py mimic

# 2. Boundary pre-training on C4
run run_boundary_pretraining.py --gpus "${gpus[@]}" --output-dir "$c4"

# 3. Fine-tuning on the news dataset
run run_news_finetuning.py --experiment backbones boundary_pretraining granularity objectives clinical_initialization \
    --gpu "$gpu" --c4-model "$c4" --output-dir "$news"

# MIND silver labels from this run's news teacher
run prepare_data.py mind --gpu "$gpu" --teacher-model "$news/modernbert_c4_1024_ts" --output-dir "$mind/data"

# 4. MIND adaptation and news fine-tuning; shared adaptations train once
run run_mind_adaptation.py --experiment scale final objective_matrix matched_steps clinical_initialization \
    --gpu "$gpu" --c4-model "$c4" --prepared-root "$mind/data/prepared" --output-dir "$mind"

# 5. Sliding-window inference over long news articles
for experiment in strides stage_matrix matched_steps baselines; do
    run run_long_document_inference.py --experiment "$experiment" --gpu "$gpu" \
        --news-models-dir "$news" --mind-models-dir "$mind" --output-dir "$inference/$experiment"
done

# 6. Transfer to MIMIC-IV discharge notes
run run_clinical_transfer.py zero-shot --gpus "$gpu" --news-ft "$news/modernbert_c4_1024_full_sentences" \
    --news-pretrain "$mind/mind_25k_cssl_news_ts" --output-dir "$clinical"
run run_clinical_transfer.py scaling --gpus "${gpus[@]}" --news-ft "$news/modernbert_c4_1024_full_sentences" \
    --news-pretrain "$mind/mind_25k_cssl_news_ts" --output-dir "$clinical"
