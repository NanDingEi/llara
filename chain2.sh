#!/bin/bash
cd /home/bbx/user_lx/LLaRA_baseline
PY=~/anaconda3/envs/minigpt4/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "chain_after_[v]3" > /dev/null; do sleep 180; done
run() {
  name=$1; shift
  echo "=== SWEEP $name start $(date +%H:%M) ==="
  $PY ours/run_ours.py --max_epochs 1 --output_dir ./output/sweep_$name/ "$@" \
    > sweep/$name.log 2>&1
  grep -E "\[epoch|score-diag" sweep/$name.log | tail -4
  echo "=== SWEEP $name done $(date +%H:%M) ==="
}
run v1e --lr 2e-4 --batch_size 4 --lora_targets qv
run v3  --lr 2e-4 --mf_lr 1e-4 --sem_align_weight 1e-4 --batch_size 4
echo "=== CHAIN2 ALL DONE $(date +%H:%M) ==="
