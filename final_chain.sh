#!/bin/bash
cd /home/bbx/user_lx/LLaRA_baseline
PY=~/anaconda3/envs/minigpt4/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "output_dir ./output/sweep_v[1]e/" > /dev/null; do sleep 120; done
run() {
  name=$1; shift
  echo "=== SWEEP $name start $(date +%H:%M) ==="
  $PY ours/$1 --max_epochs 1 --output_dir ./output/sweep_$name/ "${@:2}" \
    > sweep/$name.log 2>&1
  grep -E "\[epoch|score-diag|TEST" sweep/$name.log | tail -4
  echo "=== SWEEP $name done $(date +%H:%M) ==="
}
run llara_ml run_llara_ml.py --lr 2e-4
run v1b run_ours.py --lr 2e-4 --batch_size 4
run v1c run_ours.py --lr 1e-4 --batch_size 4
run v2  run_ours.py --lr 2e-4 --mf_lr 1e-4 --batch_size 4
run v3  run_ours.py --lr 2e-4 --mf_lr 1e-4 --sem_align_weight 1e-4 --batch_size 4
echo "=== FINAL CHAIN ALL DONE $(date +%H:%M) ==="
