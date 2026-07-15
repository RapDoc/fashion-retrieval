#!/usr/bin/env bash
# End-to-end pipeline. Assumes data/ is populated (see README).
# Indexing steps need a GPU; search + eval run on CPU.
set -e
ENC=fashionsiglip

echo "== 1. Indexing (GPU) =="
python indexer/embed_global.py  --encoder $ENC
python indexer/embed_regions.py --encoder $ENC

echo "== 2. Two-stage retrieval =="
python retriever/search_two_stage.py --encoder $ENC

echo "== 3. Cross-encoder verification (top-20) =="
python retriever/rerank_crossencoder.py --encoder $ENC --topk_ce 20

echo "== 4. Evaluation =="
python eval/run_eval.py --rankings artifacts/rankings_${ENC}_crossencoder.json \
    --name "Full system (two-stage + cross-encoder)"

echo "== 5. Visual results (top-5 per official query) =="
python eval/show_results.py --rankings artifacts/rankings_${ENC}_crossencoder.json
