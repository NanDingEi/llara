#!/bin/bash
cd /home/bbx/user_lx/LLaRA_baseline
PY=~/anaconda3/envs/minigpt4/bin/python
mkdir -p sweep
run() {
  name=$1; shift
  echo "=== SWEEP $name start $(date +%H:%M) ==="
  $PY ours/run_ours.py --max_epochs 1 --output_dir ./output/sweep_$name/ "$@" \
    > sweep/$name.log 2>&1
  grep -E "\[epoch|TEST" sweep/$name.log | tail -2
  echo "=== SWEEP $name done $(date +%H:%M) ==="
}
run v1b --lr 2e-4
run v1c --lr 1e-4
run v1d --lr 2e-4 --no_curriculum
run v2  --lr 2e-4 --mf_lr 1e-4
run v3  --lr 2e-4 --mf_lr 1e-4 --sem_align_weight 1e-4
echo "=== SWEEP ALL DONE $(date +%H:%M) ==="
