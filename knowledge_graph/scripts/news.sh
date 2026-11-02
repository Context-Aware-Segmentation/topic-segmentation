#!/usr/bin/env bash
# News KG pipeline: bash scripts/news.sh STAGE OUT (from knowledge_graph/).
# all uses saved models; reproduce includes training.
# See SETUP.md for stage dependencies and path overrides.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
stage="${1:?Usage: news.sh STAGE OUT}"
out="$(mkdir -p "${2:?Usage: news.sh STAGE OUT}" && cd "$2" && pwd)"
data="$repo_root/knowledge_graph/data"
seg_model="${SEG_MODEL:-$repo_root/topic_segmentation/runs/mind_adaptation/mind_25k_full_news_ts}"
models="${MODELS:-$out/docre_models}"
if [[ -z "${MODELS:-}" && ! -d "$models" ]]; then
    models="$repo_root/knowledge_graph/runs/news/fixed/docre_models"
fi
index_2m=sapienzanlp/relik-retriever-e5-base-v2-aida-blink-wikipedia-2Mpopular-index
index_updated=sapienzanlp/relik-retriever-e5-base-v2-aida-blink-wikipedia-index-updated
partitions=(document semantic equal random)
export TOKENIZERS_PARALLELISM=false

kg() {
    local module="knowledge_graph.$1"
    shift
    if [[ -n "${KG_PYTHON:-}" ]]; then
        PYTHONPATH="$repo_root/knowledge_graph/src:$repo_root/topic_segmentation/src" "$KG_PYTHON" -m "$module" "$@"
    else
        (source "$repo_root/knowledge_graph/scripts/activate.sh" && cd "$repo_root/knowledge_graph" && python -m "$module" "$@")
    fi
}

seg() {
    local module="$1"
    shift
    if [[ -n "${SEG_PYTHON:-}" ]]; then
        PYTHONPATH="$repo_root/topic_segmentation/src" "$SEG_PYTHON" -m "$module" "$@"
    else
        (source "$repo_root/topic_segmentation/scripts/activate.sh" && cd "$repo_root/topic_segmentation" && python -m "$module" "$@")
    fi
}

run_segment() {  # DWIE and MIND-50 sentences -> segment boundaries with the final news segmentation model
    kg news.data dwie --out-dir "$out"
    kg news.data mind-input --out "$out/mind50_segmentation_input.jsonl"
    for corpus in dwie_test mind50; do
        seg topic_segmentation.inference --model "$seg_model" --input "$out/${corpus}_segmentation_input.jsonl" \
            --output "$out/${corpus%_test}_predictions.jsonl" --window 1024 --stride 512 --protect-window-end
    done
}

run_data() {  # the DWIE QID gold and the semantic, equal, and random partitions (after segment)
    kg news.data qids --gold "$out/dwie_test_gold.json" --out "$out/dwie_test_qid_gold.json"
    kg news.data partitions --gold "$out/dwie_test_gold.json" --segments "$out/dwie_predictions.jsonl" \
        --out "$out/dwie_test_partitions.json"
}

run_inputs() {  # DocRED inputs with gold entities
    kg news.inputs gold-entities --partitions "$out/dwie_test_partitions.json" --out-dir "$out/docre"
}

run_nerl() {  # spaCy and ReLiK on DWIE partitions and on MIND-50 articles and predicted segments (GPU for ReLiK)
    for partition in "${partitions[@]}"; do
        dwie=(--input "$out/dwie_test_gold.json" --partitions "$out/dwie_test_partitions.json" --partition "$partition")
        kg news.entity_linking --backend spacy "${dwie[@]}" --out "$out/dwie_spacy_$partition.json"
        kg news.entity_linking --backend relik "${dwie[@]}" --index "$index_2m" --out "$out/dwie_relik_$partition.json"
    done
    mind=(--input "$data/mind50/mind_nerl_gold_qid.json")
    for input in document semantic; do
        segments=(); [[ $input == semantic ]] && segments=(--segments "$out/mind50_predictions.jsonl")
        kg news.entity_linking --backend spacy "${mind[@]}" "${segments[@]}" --out "$out/mind50_spacy_$input.json"
        kg news.entity_linking --backend relik "${mind[@]}" "${segments[@]}" --out "$out/mind50_relik_$input.json"
    done
    kg news.entity_linking --backend relik "${mind[@]}" --index "$index_2m" --out "$out/mind50_relik_2mpopular.json"
    kg news.entity_linking --backend relik "${mind[@]}" --index "$index_updated" --out "$out/mind50_relik_updated.json"
}

run_score() {  # NER/EL of every span policy, the ReLiK index comparison, and title-level EL
    cache="$out/prediction_title_qids.json"
    saved_cache="$repo_root/knowledge_graph/runs/news/final/prediction_title_qids.json"
    if [[ ! -e "$cache" && -e "$saved_cache" ]]; then
        cp "$saved_cache" "$cache"
    fi
    dwie=(); relik=()
    for partition in "${partitions[@]}"; do
        dwie+=(--dwie "$partition" "$out/dwie_spacy_$partition.json" "$out/dwie_relik_$partition.json")
        relik+=(--relik "$partition" "$out/dwie_relik_$partition.json")
    done
    kg news.score policies --out "$out/policies.json" --cache "$cache" \
        --mind document "$out/mind50_spacy_document.json" "$out/mind50_relik_document.json" \
        --mind semantic "$out/mind50_spacy_semantic.json" "$out/mind50_relik_semantic.json" \
        "${dwie[@]}" --dwie-gold "$out/dwie_test_qid_gold.json"
    kg news.score indexes --cache "$cache" --out "$out/indexes.json" \
        --relik default "$out/mind50_relik_document.json" --relik 2Mpopular "$out/mind50_relik_2mpopular.json" \
        --relik updated "$out/mind50_relik_updated.json"
    kg news.score titles --gold "$out/dwie_test_gold.json" "${relik[@]}" --out "$out/titles.json"
}

run_mentions() {  # DocRED inputs with the combined spaCy + ReLiK mentions as singleton entities
    kg news.inputs predicted-mentions --gold "$out/dwie_test_gold.json" --partitions "$out/dwie_test_partitions.json" \
        --spacy "$out/dwie_spacy_document.json" --relik "$out/dwie_relik_document.json" \
        --rel2id "$out/docre/rel2id.json" --out-dir "$out/predicted_docre"
}

run_train() {
    for implementation in atlop dreeam; do
        kg news.train --implementation "$implementation" --inputs "$out/docre" --out-dir "$out/docre_models" \
            --epochs "${TRAIN_EPOCHS:-10}"
    done
    models="$out/docre_models"
}

run_relations() {  # gold- and predicted-entity RE on every input condition, with the document-level row
    for implementation in atlop dreeam; do
        model=(--implementation "$implementation" --model "$models/$implementation/best.pt")
        model+=(--gold "$out/dwie_test_gold.json")
        kg news.relations predict "${model[@]}" --inputs "$out/docre" --out-dir "$out/relations_gold/$implementation"
        kg news.relations predict "${model[@]}" --inputs "$out/predicted_docre" \
            --out-dir "$out/relations_predicted/$implementation"
    done
}

case "$stage" in
    segment|data|inputs|nerl|score|mentions|train|relations) "run_$stage" ;;
    all) for s in segment data inputs nerl score mentions relations; do "run_$s"; done ;;
    reproduce) for s in segment data inputs nerl score mentions train relations; do "run_$s"; done ;;
    *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
