"""Train the LLaRA SASRec backbone directly on OUR CTR task:
score = <state_hidden(his-10), item_emb(candidate)> -> BCE(label).
The learned item embeddings become the collaborative side for injection.
Architecture = LLaRA's SASRec; only the training objective matches our task.
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


def rows(df, item_num, max_hist=10):
    out = []
    for r in df.itertuples(index=False):
        his = [i for i, t in zip(r.his, r.his_title) if t != ''][-max_hist:]
        if not his:
            continue
        states = his + [item_num] * (max_hist - len(his))
        out.append((states, len(his), int(r.iid), int(r.label), int(r.uid)))
    return out


def evaluate(model, sess, device):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for i in range(0, len(sess), 512):
            chunk = sess[i:i + 512]
            states = torch.tensor([s[0] for s in chunk], device=device)
            lens = torch.tensor([s[1] for s in chunk], device=device)
            cands = torch.tensor([s[2] for s in chunk], device=device)
            h = model.cacul_h(states, lens).squeeze(1)             # [B, 64]
            ce = model.item_embeddings(cands)                       # [B, 64]
            s = (h * ce).sum(-1).cpu().numpy()
            scores.extend(s.tolist())
            labels.extend([c[3] for c in chunk])
    model.train()
    from run_ours import global_auc
    return global_auc(np.array(scores), np.array(labels))


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda')
    sys.path.insert(0, 'ours')
    from run_ours import global_auc
    tr = pd.read_pickle(args.data_dir + 'train_ood2.pkl')
    vs = pd.read_pickle(args.data_dir + args.val_file)
    item_num = int(max(tr.iid.max(), vs.iid.max(), tr.his.explode().max())) + 1
    train_sess = rows(tr, item_num)
    val_sess = rows(vs, item_num)
    print(f'item_num={item_num} train={len(train_sess)} val={len(val_sess)}')

    model = SASRec(hidden_size=args.hidden, item_num=item_num,
                   state_size=10, dropout=args.dropout, device=device,
                   num_heads=args.num_heads).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    best = 0.0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train_sess)
        ep_loss, nb = 0.0, 0
        for i in range(0, len(train_sess), args.batch_size):
            chunk = train_sess[i:i + args.batch_size]
            states = torch.tensor([c[0] for c in chunk], device=device)
            lens = torch.tensor([c[1] for c in chunk], device=device)
            cands = torch.tensor([c[2] for c in chunk], device=device)
            labels = torch.tensor([float(c[3]) for c in chunk], device=device)
            h = model.cacul_h(states, lens).squeeze(1)             # [B, hdim]
            ce = model.item_embeddings(cands)                       # [B, hdim]
            if args.inbatch:
                scores = h @ ce.t()
                eye = torch.eye(len(chunk), dtype=torch.bool, device=device)
                collision = (cands.view(-1, 1) == cands.view(1, -1))
                valid_m = (~collision) | eye
                loss_mat = loss_fn(scores, eye.float()) * valid_m.float()
                loss = loss_mat.sum() / valid_m.float().sum().clamp_min(1)
            else:
                pos_scores = (h * ce).sum(-1)
                negs = torch.tensor(
                    [random.randint(0, item_num - 1)
                     for _ in chunk for _ in range(args.neg_per_session)],
                    device=device)
                h_rep = h.repeat_interleave(args.neg_per_session, dim=0)
                ce_rep = model.item_embeddings(negs)
                neg_scores = (h_rep * ce_rep).sum(-1)
                loss = (loss_fn(pos_scores, torch.ones_like(pos_scores)).mean()
                        + loss_fn(neg_scores, torch.zeros_like(neg_scores)).mean())
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); nb += 1
        auc = evaluate(model, val_sess, device)
        tr_auc = evaluate(model, train_sess[:5200], device)
        marker = ''
        if auc > best:
            best = auc
            torch.save(model.state_dict(), args.out + '_best_sd.pt')
            marker = ' *best*'
        print(f"ep{epoch} loss={ep_loss/max(nb,1):.4f} valid AUC={auc:.4f} "
              f"train AUC={tr_auc:.4f}{marker} ({time.time()-t0:.0f}s)", flush=True)
    torch.save(model, args.out + '.pt')
    torch.save(model.state_dict(), args.out + '_sd.pt')
    print(f'saved {args.out}.pt (best valid AUC={best:.4f})')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='/home/bbx/user_lx/GraphLoRA/dataset/ml-1m/')
    p.add_argument('--val_file', default='valid_small_ood2.pkl')
    p.add_argument('--out', default='./rec_model/sasrec_ctr.pt')
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--neg_per_session', type=int, default=1)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--num_heads', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--inbatch', action='store_true')
    p.add_argument('--mf_init', action='store_true')
    p.add_argument('--mf_path', default='/home/bbx/user_lx/GraphLoRA/checkpoints/0912_ml1m_oodv2_best_model_d256lr-0.001wd0.0001.pth')
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()
    main(args)
