#!/bin/bash
cd /home/bbx/user_lx/LLaRA_baseline
PY=~/anaconda3/envs/minigpt4/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# wait for the llara_ml chain (which follows v3) to finish
while pgrep -f "chain_llara_m[l]" > /dev/null; do sleep 180; done
run() {
  name=$1; shift
  echo "=== MULTI $name start $(date +%H:%M) ==="
  $PY ours/run_ours.py --output_dir ./output/multi_$name/ "$@" \
    > sweep/$name.log 2>&1
  grep -E "\[epoch|TEST" sweep/$name.log | tail -5
  echo "=== MULTI $name done $(date +%H:%M) ==="
}
run v1c_3ep --max_epochs 3 --lr 1e-4 --batch_size 4
run v1e_3ep --max_epochs 3 --lr 2e-4 --batch_size 4 --lora_targets qv
echo "=== MULTI ALL DONE $(date +%H:%M) ==="
