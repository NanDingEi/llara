"""LLaRA native movielens data + native SASRec side + our Yes/No task.
Everything except the task head and metrics is original LLaRA."""
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
sys.path.insert(0, 'ours')
from ours.llara_ml_yesno import LlaraMlYesNo, LlaraMlCollater
from ours.model_ours import LLaRAOurs
from optims import LinearWarmupCosineLRScheduler
from run_ours import global_auc, uauc, to_device, set_seed


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
    model.train()
    return auc, ua, users


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda')
    model = LLaRAOurs(llm_path=args.llm_path, sasrec_path=args.sasrec_path,
                      lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                      lora_dropout=args.lora_dropout, rec_size=64,
                      lora_targets=args.lora_targets).to(device)
    with open(args.prompt_path) as f:
        prompt_list = [p.strip() for p in f.read().splitlines() if p.strip()]
    datasets = {st: LlaraMlYesNo(data_dir=args.data_dir, stage=st,
                                 cans_num=args.cans_num, seed=args.seed)
                for st in ('train', 'val', 'test')}
    total_micro = max(1, len(datasets['train']) // args.batch_size) * args.max_epochs
    collate_train = LlaraMlCollater(prompt_list=prompt_list,
                                    llm_tokenizer=model.llama_tokenizer,
                                    train=True, max_step=total_micro,
                                    curriculum=not args.no_curriculum)
    collate_eval = LlaraMlCollater(prompt_list=prompt_list,
                                   llm_tokenizer=model.llama_tokenizer,
                                   train=False, curriculum=False)
    loaders = {
        'train': DataLoader(datasets['train'], batch_size=args.batch_size,
                            shuffle=True, drop_last=True,
                            num_workers=args.num_workers, collate_fn=collate_train),
        'val': DataLoader(datasets['val'], batch_size=args.eval_batch_size,
                          shuffle=False, num_workers=args.num_workers,
                          collate_fn=collate_eval),
        'test': DataLoader(datasets['test'], batch_size=args.eval_batch_size,
                           shuffle=False, num_workers=args.num_workers,
                           collate_fn=collate_eval),
    }
    print({k: len(v) for k, v in datasets.items()})
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable)/1e6:.2f}M")
    optimizer = torch.optim.Adam([
        {'params': model.projector.parameters(), 'lr': args.lr},
        {'params': model.llama_model.parameters(), 'lr': args.lr},
    ])
    micro_per_epoch = len(loaders['train'])
    max_steps = micro_per_epoch * args.max_epochs // args.accumulate_grad_batches
    scheduler = LinearWarmupCosineLRScheduler(
        optimizer, max_step=max_steps, min_lr=args.lr_decay_min_lr,
        init_lr=args.lr, warmup_steps=max_steps // 20,
        warmup_start_lr=args.lr_warmup_start_lr)
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
                    loss, l_sem = model(batch, sem_weight=0.0)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                print(f"ep{epoch} it{it+1}: OOM skipped", flush=True)
                micro += 1
                continue
            (loss / args.accumulate_grad_batches).backward()
            accum_loss += loss.item()
            n_acc += 1
            micro += 1
            if (it + 1) % args.accumulate_grad_batches == 0 or (it + 1) == len(loaders['train']):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step(micro // args.accumulate_grad_batches, epoch, max_steps)
            if (it + 1) % 1000 == 0:
                print(f"ep{epoch} it{it+1}/{micro_per_epoch} "
                      f"loss={accum_loss/max(n_acc,1):.4f} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
                accum_loss, n_acc = 0.0, 0
        auc, ua, users = evaluate(model, loaders['val'], device, tag=' valid')
        print(f"[epoch {epoch}] valid AUC={auc:.4f} session-UAUC={ua:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        results["epochs"][f"ep{epoch}"] = {"valid_auc": auc, "session_uauc": ua}
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                          if 'lora_' in k or 'projector' in k}
            torch.save(best_state, op.join(args.output_dir, 'best_state.pt'))
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    test_auc, test_ua, _ = evaluate(model, loaders['test'], device, tag=' test')
    print(f"[TEST/best] test AUC={test_auc:.4f} session-UAUC={test_ua:.4f}", flush=True)
    results["test"] = {"auc": test_auc, "session_uauc": test_ua,
                       "best_valid_auc": best_auc}
    with open(op.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='./data/ref/movielens')
    p.add_argument('--prompt_path', default='./prompt/ours.txt')
    p.add_argument('--llm_path', default='/home/bbx/user_lx/models/vicuna-7b-v0')
    p.add_argument('--sasrec_path', default='./rec_model/movielens.pt')
    p.add_argument('--output_dir', default='./output/llara_ml_yesno/')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--eval_batch_size', type=int, default=8)
    p.add_argument('--accumulate_grad_batches', type=int, default=8)
    p.add_argument('--max_epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--lr_warmup_start_lr', type=float, default=8e-6)
    p.add_argument('--lr_decay_min_lr', type=float, default=8e-6)
    p.add_argument('--lora_r', type=int, default=8)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--lora_dropout', type=float, default=0.1)
    p.add_argument('--cans_num', type=int, default=10)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--autocast', type=lambda x: x.lower() == 'true', default=True)
    p.add_argument('--no_curriculum', action='store_true')
    p.add_argument('--lora_targets', default='all', choices=['all', 'qv'])
    args = p.parse_args()
    main(args)
