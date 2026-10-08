import os.path as op
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import random


class OurOodData(data.Dataset):
    """GraphLoRA ml-1m OOD splits adapted to the LLaRA input format.

    Each sample: hybrid history (title + [HistoryEmb]) + candidate movie
    (title + [ItemEmb]) -> Yes/No. History is truncated to the last
    `max_hist` non-padding entries (padding = empty title, iid 0).
    """

    def __init__(self, data_dir=r'dataset/ml-1m', stage='train', max_hist=10, **kwargs):
        self.__dict__.update(locals())
        self.stage = stage
        fname = {'train': 'train_ood2.pkl', 'val': 'valid_ood2.pkl', 'test': 'test_ood2.pkl'}[stage]
        if stage == 'val':
            fname = kwargs.get('val_file', fname)
        if stage == 'test':
            fname = kwargs.get('test_file', fname)
        df = pd.read_pickle(op.join(data_dir, fname))
        seqs, names, lens, items, item_names, labels, uids = [], [], [], [], [], [], []
        for row in df.itertuples(index=False):
            pairs = [(i, t) for i, t in zip(row.his, row.his_title) if t != '']
            pairs = pairs[-max_hist:]
            seqs.append([p[0] for p in pairs])
            names.append([p[1] for p in pairs])
            lens.append(len(pairs))
            items.append(int(row.iid))
            item_names.append(row.title)
            labels.append(int(row.label))
            uids.append(int(row.uid))
        self.session_data = {
            'seq': seqs, 'seq_title': names, 'len_seq': lens,
            'item_id': items, 'item_name': item_names,
            'label': labels, 'uid': uids,
        }

    def __len__(self):
        return len(self.session_data['seq'])

    def __getitem__(self, i):
        return {
            'seq': self.session_data['seq'][i],
            'seq_name': self.session_data['seq_title'][i],
            'len_seq': self.session_data['len_seq'][i],
            'item_id': self.session_data['item_id'][i],
            'item_name': self.session_data['item_name'][i],
            'label': self.session_data['label'][i],
            'uid': self.session_data['uid'][i],
        }


class OurCollater:
    """LLaRA TrainCollater adapted to Yes/No: curriculum [PH]/embedding
    swap for history + target, pair tokenization for the answer tokens."""

    def __init__(self, prompt_list=None, llm_tokenizer=None, train=False,
                 terminator="\n", max_step=1, curriculum=True, inject=True):
        self.prompt_list = prompt_list
        self.llm_tokenizer = llm_tokenizer
        self.train = train
        self.terminator = terminator
        self.max_step = max_step
        self.curriculum = curriculum
        self.inject = inject
        self.cur_step = 1

    def __call__(self, batch):
        instruction = random.choice(self.prompt_list)
        inputs_text = [instruction] * len(batch)

        thresh_hold = self.cur_step / self.max_step
        p = random.random()
        use_emb = (p < thresh_hold or not self.train) if self.curriculum else True
        for i, sample in enumerate(batch):
            input_text = inputs_text[i]
            if not self.inject:
                # ablation: pure text, no collaborative tokens at all
                input_text = input_text.replace('[HistoryHere]',
                                                ", ".join(sample['seq_name']))
                input_text = input_text.replace('[TargetHere]', sample['item_name'])
                inputs_text[i] = input_text
                continue
            if '[HistoryHere]' in input_text:
                tok = '[HistoryEmb]' if use_emb else '[PH]'
                insert_prompt = ", ".join(t + f' {tok}' for t in sample['seq_name'])
                input_text = input_text.replace('[HistoryHere]', insert_prompt)
            if '[TargetHere]' in input_text:
                tok = '[ItemEmb]' if use_emb else '[PH]'
                input_text = input_text.replace('[TargetHere]',
                                                sample['item_name'] + f' {tok}')
            inputs_text[i] = input_text
        self.cur_step += 1
        flag = not use_emb  # True -> projector frozen this step (LLaRA curriculum)

        targets_text = ['Yes' if s['label'] == 1 else 'No' for s in batch]

        seqs = [s['seq'] + [0] * (10 - len(s['seq'])) for s in batch]
        if self.train:
            targets_text = [t + self.terminator for t in targets_text]
            inputs_pair = [[p_, t] for p_, t in zip(inputs_text, targets_text)]
            batch_tokens = self.llm_tokenizer(
                inputs_pair, return_tensors="pt", padding="longest",
                truncation=False, add_special_tokens=True,
                return_attention_mask=True, return_token_type_ids=True)
        else:
            batch_tokens = self.llm_tokenizer(
                inputs_text, return_tensors="pt", padding="longest",
                truncation=False, add_special_tokens=True,
                return_attention_mask=True)
        new_batch = {
            "tokens": batch_tokens,
            "seq": torch.tensor(seqs, dtype=torch.long),
            "len_seq": torch.tensor([s['len_seq'] for s in batch], dtype=torch.long),
            "item_id": torch.tensor([s['item_id'] for s in batch], dtype=torch.long),
            "label": torch.tensor([s['label'] for s in batch], dtype=torch.long),
            "uid": torch.tensor([s['uid'] for s in batch], dtype=torch.long),
        }
        if self.train:
            new_batch["flag"] = flag
        else:
            new_batch["label"] = torch.tensor([s['label'] for s in batch], dtype=torch.long)
            new_batch["uid"] = torch.tensor([s['uid'] for s in batch], dtype=torch.long)
        return new_batch
