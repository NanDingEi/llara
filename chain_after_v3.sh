#!/bin/bash
cd /home/bbx/user_lx/LLaRA_baseline
PY=~/anaconda3/envs/minigpt4/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# wait for the surviving v3 probe (PID 74018) to finish its epoch
while kill -0 74018 2>/dev/null; do sleep 120; done
echo "=== v3 exited, its results: ==="
strings sweep/v3.log | grep -E "\[epoch|score-diag" | tail -6
run() {
  name=$1; shift
  echo "=== SWEEP $name start $(date +%H:%M) ==="
  $PY ours/run_ours.py --max_epochs 1 --output_dir ./output/sweep_$name/ "$@" \
    > sweep/$name.log 2>&1
  grep -E "\[epoch|score-diag" sweep/$name.log | tail -4
  echo "=== SWEEP $name done $(date +%H:%M) ==="
}
run v1  --lr 8e-4 --batch_size 4
run v1b --lr 2e-4 --batch_size 4
run v1c --lr 1e-4 --batch_size 4
run v2  --lr 2e-4 --mf_lr 1e-4 --batch_size 4
echo "=== CHAIN ALL DONE $(date +%H:%M) ==="
