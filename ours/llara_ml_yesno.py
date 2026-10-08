"""LLaRA's native MovieLens data + native SASRec side, adapted ONLY at the
task level: next-item selection -> Yes/No with AUC metrics. Model, data
pipeline and collaborative side (SASRec item embeddings) stay original."""
import os.path as op
import random

import pandas as pd
import torch
import torch.utils.data as data


def load_id2name(data_dir):
    id2name = {}
    with open(op.join(data_dir, 'u.item'), 'r', encoding="ISO-8859-1") as f:
        for l in f.readlines():
            ll = l.strip('\n').split('|')
            name = ll[1][:-7]
            for sub in [", The", ", A", ", An"]:
                if sub in name:
                    name = sub[2:] + " " + name.replace(sub, "")
            id2name[int(ll[0]) - 1] = name
    return id2name


class LlaraMlYesNo(data.Dataset):
    """Yes/No view of LLaRA's movielens sessions.

    train: per session one sample — the true next item (label 1) or one
    sampled negative (label 0), chosen per epoch.
    val/test: per session 1 positive + (k_neg-1) seeded negatives, all
    returned; uid = session index so per-session AUC is computable.
    """

    def __init__(self, data_dir=r'data/ref/movielens', stage='train',
                 cans_num=10, seed=1234, **kwargs):
        self.stage = stage
        self.cans_num = cans_num
        self.id2name = load_id2name(data_dir)
        fname = {'train': 'train_data.df', 'val': 'Val_data.df',
                 'test': 'Test_data.df'}[stage]
        df = pd.read_pickle(op.join(data_dir, fname))
        df = df[df['len_seq'] >= 3].reset_index(drop=True)
        self.rng = random.Random(seed)
        self.samples = []
        for i, r in df.iterrows():
            # raw df stores (item_id, rating) tuples; final id lists are
            # extracted here (their MovielensData does the same at load)
            seq = [x[0] if isinstance(x, (tuple, list)) else x for x in r['seq']]
            nxt = r['next']
            nxt = nxt[0] if isinstance(nxt, (tuple, list)) else nxt
            titles = [self.id2name.get(x, '') for x in seq if x != 1682]
            seq_ids = [x for x in seq if x != 1682]
            if not titles:
                continue
            pos = int(nxt)
            hist_set = set(seq_ids) | {pos}
            if stage == 'train':
                neg = self.rng.choice([x for x in self.id2name if x not in hist_set])
                pick = [(pos, 1)] if self.rng.random() < 0.5 else [(neg, 0)]
                for it, lb in pick:
                    self.samples.append({'seq': seq_ids, 'seq_name': titles,
                                         'len_seq': len(seq_ids),
                                         'item_id': it,
                                         'item_name': self.id2name[it],
                                         'label': lb, 'uid': i})
            else:
                negs = self.rng.sample([x for x in self.id2name if x not in hist_set],
                                       cans_num - 1)
                for it in [pos] + negs:
                    self.samples.append({'seq': seq_ids, 'seq_name': titles,
                                         'len_seq': len(seq_ids),
                                         'item_id': it,
                                         'item_name': self.id2name[it],
                                         'label': 1 if it == pos else 0,
                                         'uid': i})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


class LlaraMlCollater:
    """Same contract as OurCollater but pads the SASRec seq with its own
    padding id (1682) and keeps 10 slots."""

    def __init__(self, prompt_list=None, llm_tokenizer=None, train=False,
                 terminator="\n", max_step=1, curriculum=True, pad_id=1682):
        self.prompt_list = prompt_list
        self.llm_tokenizer = llm_tokenizer
        self.train = train
        self.terminator = terminator
        self.max_step = max_step
        self.curriculum = curriculum
        self.pad_id = pad_id
        self.cur_step = 1

    def __call__(self, batch):
        instruction = random.choice(self.prompt_list)
        inputs_text = [instruction] * len(batch)
        thresh_hold = self.cur_step / self.max_step
        p = random.random()
        use_emb = (p < thresh_hold or not self.train) if self.curriculum else True
        for i, s in enumerate(batch):
            t = inputs_text[i]
            if '[HistoryHere]' in t:
                tok = '[HistoryEmb]' if use_emb else '[PH]'
                t = t.replace('[HistoryHere]',
                              ", ".join(x + f' {tok}' for x in s['seq_name']))
            if '[TargetHere]' in t:
                tok = '[ItemEmb]' if use_emb else '[PH]'
                t = t.replace('[TargetHere]', s['item_name'] + f' {tok}')
            inputs_text[i] = t
        self.cur_step += 1
        flag = not use_emb
        targets_text = ['Yes' if s['label'] == 1 else 'No' for s in batch]
        seqs = [s['seq'] + [self.pad_id] * (10 - len(s['seq'])) for s in batch]
        if self.train:
            targets_text = [t + self.terminator for t in targets_text]
            tokens = self.llm_tokenizer(
                [[p_, t] for p_, t in zip(inputs_text, targets_text)],
                return_tensors="pt", padding="longest", truncation=False,
                add_special_tokens=True, return_attention_mask=True,
                return_token_type_ids=True)
        else:
            tokens = self.llm_tokenizer(
                inputs_text, return_tensors="pt", padding="longest",
                truncation=False, add_special_tokens=True,
                return_attention_mask=True)
        out = {"tokens": tokens,
               "seq": torch.tensor(seqs, dtype=torch.long),
               "len_seq": torch.tensor([min(s['len_seq'], 10) for s in batch],
                                       dtype=torch.long),
               "item_id": torch.tensor([s['item_id'] for s in batch],
                                       dtype=torch.long),
               "label": torch.tensor([s['label'] for s in batch], dtype=torch.long),
               "uid": torch.tensor([s['uid'] for s in batch], dtype=torch.long)}
        if self.train:
            out["flag"] = flag
        return out
