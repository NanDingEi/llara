#!/bin/bash
cd /home/bbx/user_lx/LLaRA_baseline
PY=~/anaconda3/envs/minigpt4/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p sweep final
# 5 epochs, per-epoch checkpoints, per-epoch test eval
echo "=== MF side 5ep start $(date +%H:%M) ==="
$PY ours/run_ours.py --max_epochs 5 --lr 1e-4 --batch_size 4 --save_epoch_states \
  --val_file valid_small_ood2.pkl \
  --output_dir /home/bbx/user_lx/LLaRA_baseline/output/final_mf_5ep/ \
  > final/mf_5ep.log 2>&1
grep -E "\[TEST ep|\[TEST/best|\[epoch" final/mf_5ep.log | tail -12
echo "=== MF side 5ep done $(date +%H:%M) ==="
echo "=== SASRec side 5ep start $(date +%H:%M) ==="
$PY ours/run_ours.py --max_epochs 5 --lr 1e-4 --batch_size 4 --save_epoch_states \
  --sasrec_path /home/bbx/user_lx/LLaRA_baseline/rec_model/sasrec_ctr_best.pt \
  --emb_scale 0.089 --val_file valid_small_ood2.pkl \
  --output_dir /home/bbx/user_lx/LLaRA_baseline/output/final_sas_5ep/ \
  > final/sas_5ep.log 2>&1
grep -E "\[TEST ep|\[TEST/best|\[epoch" final/sas_5ep.log | tail -12
echo "=== SASRec side 5ep done $(date +%H:%M) ==="
$PY ours/summarize_final.py || true
echo "=== FINAL RUNS ALL DONE $(date +%H:%M) ==="
