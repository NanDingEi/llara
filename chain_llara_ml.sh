#!/bin/bash
cd /home/bbx/user_lx/LLaRA_baseline
PY=~/anaconda3/envs/minigpt4/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "output_dir ./output/sweep_v[3]/" > /dev/null; do sleep 180; done
echo "=== SWEEP llara_ml start $(date +%H:%M) ==="
$PY ours/run_llara_ml.py --max_epochs 1 --lr 2e-4 --output_dir ./output/sweep_llara_ml/ \
  > sweep/llara_ml.log 2>&1
grep -E "\[epoch|score-diag|TEST" sweep/llara_ml.log | tail -4
echo "=== SWEEP llara_ml done $(date +%H:%M) ==="
