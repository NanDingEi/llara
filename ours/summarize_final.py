"""Merge the two 5-epoch runs' results into one summary file."""
import json
import os.path as op

base = '/home/bbx/user_lx/LLaRA_baseline'
runs = {
    'MF_side_5ep': op.join(base, 'output/final_mf_5ep/results.json'),
    'SASRec_side_5ep': op.join(base, 'output/final_sas_5ep/results.json'),
}
summary = {}
for name, path in runs.items():
    if op.exists(path):
        with open(path) as f:
            summary[name] = json.load(f)

lines = ['# 5-epoch final comparison (valid_small 5200 / test_ood2 7331)', '']
for name, res in summary.items():
    lines.append(f'## {name}')
    lines.append('| epoch | valid AUC | valid UAUC | test AUC | test UAUC |')
    lines.append('|---|---|---|---|---|')
    for ep, rec in res.get('per_epoch_test', {}).items():
        lines.append(f"| {ep} | {rec['valid_auc']:.4f} | {rec['valid_uauc']:.4f} "
                     f"| {rec['test_auc']:.4f} | {rec['test_uauc']:.4f} |")
    t = res.get('test', {})
    lines.append(f"| best-valid | {t.get('best_valid_auc', 0):.4f} | — "
                 f"| {t.get('auc', 0):.4f} | {t.get('uauc', 0):.4f} |")
    lines.append('')
with open(op.join(base, 'final/comparison_5ep.md'), 'w') as f:
    f.write('\n'.join(lines))
with open(op.join(base, 'final/comparison_5ep.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print('summary written: final/comparison_5ep.md')
