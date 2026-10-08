"""Train LLaRA's native SASRec on the GraphLoRA ml-1m OOD data.

Faithful to the LLaRA SASRec recipe: hidden 64, state_size 10, BCE with one
sampled negative per session, Adam. Data: train_ood2.pkl sessions
(his[-10:] -> iid). Evaluates HR@10/NDCG@10 on valid_ood2 next-item.
"""
import argparse
import os
import sys
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recommender.A_SASRec_final_bce_llm import SASRec


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_corpus(ratings_path, users, tmin_m, item_num, max_hist=10):
    """LLaRA-style corpus from the RAW log: per-user chronological positive
    interactions; next-item transitions. Includes the PRE-WINDOW months
    (no task labels exist there — pure sequence pretraining) plus the
    train-window positives."""
    r = pd.read_csv(ratings_path, sep='::', engine='python',
                    names=['uid', 'iid', 'rating', 'ts'])
    r = r[r.uid.isin(users) & (r.rating >= 4)].copy()
    r['m'] = pd.to_datetime(r.ts, unit='s').dt.year * 12 + pd.to_datetime(r.ts, unit='s').dt.month
    r = r.sort_values(['uid', 'ts'])
    sessions = []
    for u, g in r.groupby('uid'):
        seq = [i for i in g.iid.tolist() if i < item_num]
        # dedupe consecutive repeats
        ded = [seq[0]] if seq else []
        for x in seq[1:]:
            if x != ded[-1]:
                ded.append(x)
        for j in range(1, len(ded)):
            his = ded[max(0, j - max_hist):j]
            states = his + [item_num] * (max_hist - len(his))
            # skip transitions whose TARGET falls in the valid/test windows:
            # only train on targets strictly before the valid window start
            sessions.append((states, len(his), ded[j], g.m.iloc[j]))
    return sessions


def build_sessions(df, item_num, max_hist=10, pos_only=True):
    """sessions: (states [L<=10] padded with item_num, len, target).

    pos_only: keep only label==1 rows — for label==0 rows the target iid is
    a SAMPLED NEGATIVE (user never watched it); training next-item on those
    poisons the model (the 0.0178 HR@10 bug)."""
    out = []
    for r in df.itertuples(index=False):
        if pos_only and hasattr(r, 'label') and int(r.label) != 1:
            continue
        his = [i for i, t in zip(r.his, r.his_title) if t != ''][-max_hist:]
        if not his:
            continue
        states = his + [item_num] * (max_hist - len(his))
        out.append((states, len(his), int(r.iid)))
    return out


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda')
    tr = pd.read_pickle(args.data_dir + 'train_ood2.pkl')
    va = pd.read_pickle(args.data_dir + 'valid_ood2.pkl')
    item_num = int(max(tr.iid.max(), va.iid.max(), tr.his.explode().max())) + 1
    print(f'item_num={item_num}')
    users = set(tr.uid.unique())
    tmin_m = int(pd.Timestamp(tr.timestamp.min(), unit='s').year * 12
                 + pd.Timestamp(tr.timestamp.min(), unit='s').month)
    vm = va.copy()
    vm['m'] = (pd.to_datetime(vm.timestamp, unit='s').dt.year * 12
               + pd.to_datetime(vm.timestamp, unit='s').dt.month)
    valid_start_m = int(vm['m'].min())
    print(f'train window starts at month {tmin_m}, valid at {valid_start_m}')
    full = build_corpus(args.ratings_path, users, tmin_m, item_num, args.max_hist)
    # corpus transitions: pre-window targets only for pretraining; the
    # train-window positive transitions come from the OOD train rows
    pre = [s for s in full if s[3] < tmin_m]
    train_sess = [(s[0], s[1], s[2]) for s in pre]
    train_sess += [(s[0], s[1], s[2]) for s in build_sessions(tr, item_num, pos_only=True)]
    valid_sess = [(s[0], s[1], s[2]) for s in build_sessions(va, item_num, pos_only=True)]
    print(f'corpus: pre-window transitions={len(pre)}, total train={len(train_sess)}, valid={len(valid_sess)}')
    print(f'sessions: train={len(train_sess)} valid={len(valid_sess)}')

    model = SASRec(hidden_size=64, item_num=item_num, state_size=10,
                   dropout=0.2, device=device).to(device)
    if args.tie:
        # weight tying: s_fc.weight shares item_embeddings.weight[0:item_num]
        # so ranking knowledge MUST live in the embedding table (the LLaRA
        # injection taps exactly this table, which otherwise stays ~random)
        model.s_fc.weight = torch.nn.Parameter(model.item_embeddings.weight[:item_num])
        print('s_fc weight TIED to item_embeddings')
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=args.lr, betas=(0.9, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()

    def eval_hr(model, sessions, k=10):
        model.eval()
        hits, ndcgs = [], []
        with torch.no_grad():
            for i in range(0, len(sessions), 256):
                chunk = sessions[i:i + 256]
                states = torch.tensor([s[0] for s in chunk], device=device)
                lens = torch.tensor([s[1] for s in chunk], device=device)
                tgts = torch.tensor([s[2] for s in chunk], device=device)
                logits = model.forward_eval(states, lens)          # [B, item_num]
                # mask out history items (standard protocol)
                for b, sess in enumerate(chunk):
                    for x in sess[0]:
                        if x != item_num:  # skip padding id
                            logits[b, x] = -1e9
                topk = logits.topk(k, dim=-1).indices
                for b, sess in enumerate(chunk):
                    t = int(sess[2])
                    rank = (topk[b] == t).nonzero()
                    hits.append(1.0 if rank.numel() else 0.0)
                    ndcgs.append(1.0 / np.log2(rank.item() + 2) if rank.numel() else 0.0)
        model.train()
        return float(np.mean(hits)), float(np.mean(ndcgs))

    best_hr = 0.0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train_sess)
        ep_loss, nb = 0.0, 0
        for i in range(0, len(train_sess), args.batch_size):
            chunk = train_sess[i:i + args.batch_size]
            states = torch.tensor([s[0] for s in chunk], device=device)
            lens = torch.tensor([s[1] for s in chunk], device=device)
            tgts = torch.tensor([s[2] for s in chunk], device=device)
            logits = model(states, lens)                            # [B, item_num]
            pos = logits.gather(1, tgts.view(-1, 1)).squeeze(1)
            # one sampled negative per session (LLaRA BCE recipe)
            negs = torch.tensor(
                [random.randint(0, item_num - 1) for _ in chunk], device=device)
            neg = logits.gather(1, negs.view(-1, 1)).squeeze(1)
            labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
            loss = loss_fn(torch.cat([pos, neg]), labels)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); nb += 1
        hr, ndcg = eval_hr(model, valid_sess)
        print(f"ep{epoch} loss={ep_loss/max(nb,1):.4f} valid HR@10={hr:.4f} "
              f"NDCG@10={ndcg:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if hr > best_hr:
            best_hr = hr
            torch.save(model.state_dict(), args.out + '_best_sd.pt')
    # save full module (loadable like LLaRA's own rec_model)
    torch.save(model, args.out + '.pt')
    torch.save(model.state_dict(), args.out + '_sd.pt')
    print(f'saved {args.out}.pt (best valid HR@10={best_hr:.4f})')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='/home/bbx/user_lx/GraphLoRA/dataset/ml-1m/')
    p.add_argument('--out', default='./rec_model/sasrec_ours.pt')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--ratings_path', default='/home/bbx/user_lx/GraphLoRA/dataset/ml-1m/ratings.dat')
    p.add_argument('--max_hist', type=int, default=10)
    p.add_argument('--tie', action='store_true')
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()
    main(args)
