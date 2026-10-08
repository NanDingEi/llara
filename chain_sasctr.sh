#!/bin/bash
cd /home/bbx/user_lx/LLaRA_baseline
PY=~/anaconda3/envs/minigpt4/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "output_dir /home/bbx/user_lx/LLaRA_baseline/output/multi_sas_3e[p]/" > /dev/null; do sleep 180; done
echo "=== sasctr_3ep start $(date +%H:%M) ==="
$PY ours/run_ours.py --max_epochs 3 --lr 1e-4 --batch_size 4 \
  --sasrec_path /home/bbx/user_lx/LLaRA_baseline/rec_model/sasrec_ctr_best.pt \
  --val_file valid_small_ood2.pkl \
  --output_dir /home/bbx/user_lx/LLaRA_baseline/output/multi_sasctr_3ep/ \
  > /home/bbx/user_lx/LLaRA_baseline/sweep/sasctr_3ep.log 2>&1
grep -E "\[epoch|\[TEST" /home/bbx/user_lx/LLaRA_baseline/sweep/sasctr_3ep.log | tail -5
echo "=== sasctr_3ep done $(date +%H:%M) ==="
