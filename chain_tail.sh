#!/bin/bash
cd /home/bbx/user_lx/LLaRA_baseline
PY=~/anaconda3/envs/minigpt4/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "output_dir ./output/multi_v1c_3e[p]/" > /dev/null; do sleep 180; done
echo "=== v1e_3ep start $(date +%H:%M) ==="
$PY ours/run_ours.py --max_epochs 3 --lr 2e-4 --batch_size 4 --lora_targets qv \
  --output_dir ./output/multi_v1e_3ep/ > sweep/v1e_3ep.log 2>&1
grep -E "\[epoch|TEST" sweep/v1e_3ep.log | tail -5
echo "=== v1e_3ep done $(date +%H:%M) ==="
echo "=== llara_ml start $(date +%H:%M) ==="
$PY ours/run_llara_ml.py --max_epochs 1 --lr 2e-4 --output_dir ./output/sweep_llara_ml/ \
  > sweep/llara_ml.log 2>&1
grep -E "\[epoch|score-diag|TEST" sweep/llara_ml.log | tail -4
echo "=== llara_ml done $(date +%H:%M) ==="
echo "=== TAIL CHAIN ALL DONE $(date +%H:%M) ==="
