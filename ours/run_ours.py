"""Train/eval LLaRA (adapted) on the GraphLoRA ml-1m OOD Yes/No task.

Faithful to the LLaRA recipe: LLaMA-7B + LoRA (r8/a32/d0.1, all-proj),
MlpProjector injecting frozen collaborative item embeddings at
[HistoryEmb]/[ItemEmb] input positions, curriculum prompting (early steps
use [PH] with the projector frozen), Adam + LinearWarmupCosine.

Metrics: AUC + user-paired UAUC (GraphLoRA protocol) on valid each epoch;
test is evaluated once with the best-valid checkpoint.
"""
import argparse
import json
import os
import os.path as op
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ours.data_ours import OurOodData, OurCollater
from ours.model_ours import LLaRAOurs
from optims import LinearWarmupCosineLRScheduler


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif hasattr(v, 'to'):  # BatchEncoding
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def global_auc(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    sorted_s = scores[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def uauc(scores, labels, uids):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    uids = np.asarray(uids)
    per_user, users_used = [], []
    for u in np.unique(uids):
        m = uids == u
        y = labels[m]
        if y.min() == y.max():
            continue
        per_user.append(global_auc(scores[m], y))
        users_used.append(int(u))
    return float(np.mean(per_user)), users_used


def evaluate(model, loader, device, tag=""):
    model.eval()
    scores, labels, uids = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            s = model.score(batch)
            scores.extend(s.tolist())
            labels.extend(batch["label"].cpu().tolist())
            uids.extend(batch["uid"].cpu().tolist())
    auc = global_auc(scores, labels)
    ua, users = uauc(scores, labels, uids)
    # degenerate-output diagnostics: p(Yes) = sigmoid(margin) stats
    pyes = 1.0 / (1.0 + np.exp(-np.asarray(scores)))
    print(f"[score-diag{tag}] p(Yes): mean={pyes.mean():.4f} "
          f"min={pyes.min():.4f} max={pyes.max():.4f} std={pyes.std():.4f}", flush=True)
    model.train()
    return auc, ua, users


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda')
    model = LLaRAOurs(llm_path=args.llm_path, mf_path=args.mf_path,
                      lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                      lora_dropout=args.lora_dropout, rec_size=256,
                      train_mf=args.mf_lr > 0,
                      lora_targets=args.lora_targets,
                      sasrec_path=args.sasrec_path,
                      freeze_projector=args.freeze_projector,
                      emb_scale=args.emb_scale).to(device)

    with open(args.prompt_path) as f:
        prompt_list = [p.strip() for p in f.read().splitlines() if p.strip()]

    datasets = {st: OurOodData(data_dir=args.data_dir, stage=st, max_hist=args.max_hist,
                               val_file=args.val_file, test_file=args.test_file)
                for st in ('train', 'val', 'test')}
    total_micro = max(1, len(datasets['train']) // args.batch_size) * args.max_epochs
    collate_train = OurCollater(prompt_list=prompt_list, llm_tokenizer=model.llama_tokenizer,
                                train=True, max_step=total_micro,
                                curriculum=not args.no_curriculum,
                                inject=not args.no_inject)
    collate_eval = OurCollater(prompt_list=prompt_list, llm_tokenizer=model.llama_tokenizer,
                               train=False, curriculum=False, inject=not args.no_inject)
    train_diag = torch.utils.data.Subset(datasets['train'], list(range(5200)))
    loaders = {
        'train': DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True,
                            drop_last=True, num_workers=args.num_workers,
                            collate_fn=collate_train),
        'train_diag': DataLoader(train_diag, batch_size=args.eval_batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=collate_eval),
        'val': DataLoader(datasets['val'], batch_size=args.eval_batch_size, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate_eval),
        'test': DataLoader(datasets['test'], batch_size=args.eval_batch_size, shuffle=False,
                           num_workers=args.num_workers, collate_fn=collate_eval),
    }

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"trainable params: {n_train/1e6:.2f}M")
    groups = [
        {'params': model.projector.parameters(), 'lr': args.lr},
        {'params': model.llama_model.parameters(), 'lr': args.lr},
    ]
    if args.mf_lr > 0:
        groups.append({'params': model.item_table.parameters(), 'lr': args.mf_lr})
        print(f"[v2/v3] MF table trainable at lr={args.mf_lr}")
    if args.sem_align_weight > 0:
        print(f"[v3] semantic relation loss ON: weight={args.sem_align_weight} layer={args.sem_layer}")
    optimizer = torch.optim.Adam(groups)
    micro_per_epoch = len(loaders['train'])
    max_steps = micro_per_epoch * args.max_epochs // args.accumulate_grad_batches
    warmup_steps = max_steps // 20
    scheduler = LinearWarmupCosineLRScheduler(
        optimizer, max_step=max_steps, min_lr=args.lr_decay_min_lr,
        init_lr=args.lr, warmup_steps=warmup_steps,
        warmup_start_lr=args.lr_warmup_start_lr)
    print(f"micro/epoch={micro_per_epoch} updates/epoch={micro_per_epoch//args.accumulate_grad_batches} max_updates={max_steps}")

    os.makedirs(args.output_dir, exist_ok=True)
    best_auc, best_state = -1.0, None
    results = {"args": vars(args), "epochs": {}}
    micro = 0
    for epoch in range(args.max_epochs):
        model.train()
        t0 = time.time()
        accum_loss, n_acc = 0.0, 0
        for it, batch in enumerate(loaders['train']):
            batch = to_device(batch, device)
            try:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                    enabled=args.autocast):
                    loss, l_sem = model(batch, sem_weight=args.sem_align_weight,
                                        sem_layer=args.sem_layer)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                print(f"ep{epoch} it{it+1}: OOM skipped", flush=True)
                micro += 1
                continue
            # model returns CE + sem_weight*L_sem already; l_sem is detached
            (loss / args.accumulate_grad_batches).backward()
            accum_loss += loss.item() / args.accumulate_grad_batches
            if l_sem is not None:
                sem_accum = getattr(main, '_sem_accum', 0.0) + l_sem.item()
                main._sem_accum = sem_accum
            n_acc += 1
            micro += 1
            if (it + 1) % args.accumulate_grad_batches == 0 or (it + 1) == len(loaders['train']):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step(micro // args.accumulate_grad_batches, epoch, max_steps)
            if (it + 1) % 500 == 0:
                _sem_msg = ''
                if args.sem_align_weight > 0:
                    _sv = getattr(main, '_sem_accum', 0.0) / max(n_acc, 1)
                    _sem_msg = f" L_sem={_sv:.5f}"
                    main._sem_accum = 0.0
                print(f"ep{epoch} it{it+1}/{micro_per_epoch} loss={accum_loss/max(n_acc,1):.4f}"
                      f"{_sem_msg} elapsed={time.time()-t0:.0f}s", flush=True)
                accum_loss, n_acc = 0.0, 0
        auc, ua, users = evaluate(model, loaders['val'], device, tag=' valid')
        tr_auc, tr_ua, _ = evaluate(model, loaders['train_diag'], device, tag=' train')
        print(f"[epoch {epoch}] valid AUC={auc:.4f} UAUC={ua:.4f} | "
              f"train-subset AUC={tr_auc:.4f} UAUC={tr_ua:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if args.save_epoch_states:
            ep_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                        if 'lora_' in k or 'projector' in k or 'item_table' in k}
            torch.save(ep_state, op.join(args.output_dir, f'epoch{epoch}_state.pt'))
        results["epochs"][f"ep{epoch}"] = {"valid_auc": auc, "valid_uauc": ua,
                                           "train_diag_auc": tr_auc,
                                           "n_uauc_users": len(users)}
        if auc > best_auc:
            best_auc = auc
            best_epoch_saved = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                          if 'lora_' in k or 'projector' in k or 'item_table' in k}
            torch.save(best_state, op.join(args.output_dir, 'best_state.pt'))
            with open(op.join(args.output_dir, 'best_epoch.txt'), 'w') as f:
                f.write(str(epoch))

    # test with best-valid weights
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    test_auc, test_ua, test_users = evaluate(model, loaders['test'], device, tag=' test')
    print(f"[TEST/best] test AUC={test_auc:.4f} UAUC={test_ua:.4f} "
          f"n_users={len(test_users)}", flush=True)
    results["test"] = {"auc": test_auc, "uauc": test_ua, "n_uauc_users": len(test_users),
                       "best_valid_auc": best_auc}
    with open(op.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("saved results.json")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='/home/bbx/user_lx/GraphLoRA/dataset/ml-1m')
    p.add_argument('--prompt_path', default='./prompt/ours.txt')
    p.add_argument('--llm_path', default='/home/bbx/user_lx/models/vicuna-7b-v0')
    p.add_argument('--mf_path', default='/home/bbx/user_lx/GraphLoRA/checkpoints/0912_ml1m_oodv2_best_model_d256lr-0.001wd0.0001.pth')
    p.add_argument('--output_dir', default='./output/ours/')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--eval_batch_size', type=int, default=8)
    p.add_argument('--accumulate_grad_batches', type=int, default=8)
    p.add_argument('--max_epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=8e-4)
    p.add_argument('--lr_warmup_start_lr', type=float, default=8e-6)
    p.add_argument('--lr_decay_min_lr', type=float, default=8e-6)
    p.add_argument('--lora_r', type=int, default=8)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--lora_dropout', type=float, default=0.1)
    p.add_argument('--max_hist', type=int, default=10)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--autocast', type=lambda x: x.lower() == 'true', default=True)
    p.add_argument('--mf_lr', type=float, default=0.0,
                   help='>0: unfreeze the MF item table with this lr (v2/v3)')
    p.add_argument('--sem_align_weight', type=float, default=0.0,
                   help='>0: GraphLoRA-style relation-MSE semantic loss (v3)')
    p.add_argument('--sem_layer', type=int, default=8)
    p.add_argument('--no_curriculum', action='store_true')
    p.add_argument('--lora_targets', default='all', choices=['all', 'qv'])
    p.add_argument('--no_inject', action='store_true',
                   help='ablation: pure text prompt, no collaborative embedding injection')
    p.add_argument('--freeze_projector', action='store_true')
    p.add_argument('--emb_scale', type=float, default=None)
    p.add_argument('--save_epoch_states', action='store_true')
    p.add_argument('--val_file', default='valid_ood2.pkl')
    p.add_argument('--test_file', default='test_ood2.pkl')
    p.add_argument('--sasrec_path', default=None,
                   help='set to use a SASRec checkpoint as the collaborative side')
    args = p.parse_args()
    main(args)
