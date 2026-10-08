import torch
from torch import nn
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import get_peft_model, LoraConfig, TaskType

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.mlp_projector import MlpProjector
from optims import LinearWarmupCosineLRScheduler


class LLaRAOurs(nn.Module):
    """LLaRA core (LLaMA + LoRA + input-side collaborative embedding
    injection via [HistoryEmb]/[ItemEmb] special tokens + MlpProjector)
    adapted to the GraphLoRA Yes/No task with AUC/UAUC metrics.

    The collaborative side is the pretrained MF item table (256-d, frozen).
    """

    def __init__(self, llm_path, mf_path=None, lora_r=8, lora_alpha=32,
                 lora_dropout=0.1, rec_size=256, train_mf=False,
                 lora_targets='all', sasrec_path=None, freeze_projector=False,
                 emb_scale=None):
        super().__init__()
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llm_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llama_tokenizer.padding_side = "right"
        self.llama_tokenizer.add_special_tokens(
            {'additional_special_tokens': ['[PH]', '[HistoryEmb]', '[CansEmb]', '[ItemEmb]']})
        self.llama_model = LlamaForCausalLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16)
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        targets = (['k_proj', 'v_proj', 'q_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj']
                   if lora_targets == 'all' else ['q_proj', 'v_proj'])
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                                 r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                 target_modules=targets)
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.print_trainable_parameters()
        # 24GB card: long history prompts (10 items, ~800 tokens) OOM at bs2
        # without checkpointing; non-reentrant works with frozen embeddings
        try:
            # old transformers: no gradient_checkpointing_kwargs kwarg.
            # Reentrant ckpt is safe here WITHOUT enable_input_require_grads:
            # in training our inputs_embeds already requires grad (projector
            # outputs are index-assigned into it), and eval skips the ckpt
            # branch entirely (self.training is False).
            self.llama_model.gradient_checkpointing_enable()
            # reentrant ckpt needs SOME input with requires_grad; PH-phase
            # batches have no injected embeddings (pure frozen token lookup),
            # so force the embedding output to require grad during training
            def _make_inputs_require_grads(module, inputs, output):
                if torch.is_grad_enabled():
                    output.requires_grad_(True)
            self.llama_model.get_input_embeddings().register_forward_hook(
                _make_inputs_require_grads)
            print('gradient checkpointing enabled (legacy reentrant)')
        except Exception as e:
            print('gradient checkpointing NOT enabled:', e)
        print('Loading LLAMA Done')

        # collaborative side: pretrained MF item table. Frozen (v1) or
        # trainable (v2/v3: gradient backprop from the rec loss and/or the
        # semantic-relation loss reaches the embeddings)
        # collaborative side: either the GraphLoRA MF table (256-d) or
        # LLaRA's native SASRec item embeddings (64-d, cacu_x = pure lookup,
        # so both share the same interface)
        if sasrec_path is not None:
            import importlib
            mod = importlib.import_module('recommender.A_SASRec_final_bce_llm')
            sys_modules_SASRec = sys.modules['__main__']
            if not hasattr(sys_modules_SASRec, 'SASRec'):
                sys_modules_SASRec.SASRec = mod.SASRec
            sas = torch.load(sasrec_path, map_location="cpu", weights_only=False)
            sas.eval()
            for p in sas.parameters():
                p.requires_grad_(False)
            self.item_table = sas.item_embeddings
            self.rec_size = self.item_table.embedding_dim
            # normalize the SASRec table to the MF-table scale (~0.71 norm):
            # raw N(0,1)-init rows have norm ~8, which makes the projector's
            # gradients 11x larger and destabilizes training into the
            # memorization attractor
            if emb_scale is not None and emb_scale > 0:
                with torch.no_grad():
                    self.item_table.weight.mul_(emb_scale)
                print(f'[emb-scale] SASRec table scaled x{emb_scale} '
                      f'(new norm mean={self.item_table.weight.norm(dim=1).mean():.3f})')
            print(f'Loading SASRec side: item_num={sas.item_num} dim={self.rec_size}')
        else:
            print('Loading MF side info (train_mf=%s)' % train_mf)
            mf = torch.load(mf_path, map_location="cpu")
            item_emb = mf['item_embedding.weight'].float()
            self.item_table = nn.Embedding(item_emb.shape[0], item_emb.shape[1])
            self.item_table.weight.data.copy_(item_emb)
            self.item_table.weight.requires_grad_(bool(train_mf))
            self.rec_size = item_emb.shape[1]

        self.projector = MlpProjector(rec_size=self.rec_size,
                                      llm_size=self.llama_model.config.hidden_size)
        # freeze_projector (diagnostic): keep the random-init projector
        # frozen so the injection channel cannot become a train-label
        # decoder (the probe showed the trained projector drives train AUC
        # to 0.92 while valid degrades — memorization at the entry of the
        # injection path)
        self.freeze_projector = bool(freeze_projector)
        if self.freeze_projector:
            for p in self.projector.parameters():
                p.requires_grad_(False)
            print('[diag] projector FROZEN at random init', flush=True)

        self.yes_id = self.llama_tokenizer('Yes', add_special_tokens=False).input_ids[0]
        self.no_id = self.llama_tokenizer('No', add_special_tokens=False).input_ids[0]
        print(f'answer token ids: yes={self.yes_id} no={self.no_id}')

    def encode_items(self, seq):
        item_rec_embs = self.item_table(seq)      # [B, L, 256] (frozen MF)
        return self.projector(item_rec_embs)      # [B, L, llm_size]

    def wrap_emb(self, batch, return_positions=False):
        input_embeds = self.llama_model.get_input_embeddings()(batch["tokens"].input_ids)
        his_token_id = self.llama_tokenizer("[HistoryEmb]", return_tensors="pt",
                                            add_special_tokens=False).input_ids.item()
        item_token_id = self.llama_tokenizer("[ItemEmb]", return_tensors="pt",
                                             add_special_tokens=False).input_ids.item()
        his_item_embeds = self.encode_items(batch["seq"])
        item_embeds = self.encode_items(batch["item_id"])
        B = len(batch["len_seq"])
        his_pos = torch.full((B, batch["seq"].shape[1]), -1, dtype=torch.long)
        tgt_pos = torch.full((B,), -1, dtype=torch.long)
        rows, cols, vals = [], [], []
        for i in range(B):
            if (batch["tokens"].input_ids[i] == his_token_id).nonzero().shape[0] > 0:
                idx_tensor = (batch["tokens"].input_ids[i] == his_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor, his_item_embeds[i, :batch["len_seq"][i].item()]):
                    rows.append(i); cols.append(idx.item()); vals.append(item_emb)
                his_pos[i, :batch["len_seq"][i].item()] = idx_tensor.cpu()
            idx = (batch["tokens"].input_ids[i] == item_token_id).nonzero()
            if idx.shape[0] > 0:
                p_ = idx.view(-1).item()
                rows.append(i); cols.append(p_); vals.append(item_embeds[i])
                tgt_pos[i] = p_
        if rows:
            # OUT-OF-PLACE index_put: in-place assignment on the (possibly
            # leaf, grad-forced) embedding output breaks autograd
            input_embeds = input_embeds.index_put(
                (torch.tensor(rows, device=input_embeds.device),
                 torch.tensor(cols, device=input_embeds.device)),
                torch.stack(vals).to(input_embeds.dtype))
        return input_embeds, his_pos, tgt_pos

    def semantic_loss(self, hidden_states, batch, his_pos, tgt_pos, layer=8):
        """GraphLoRA-style L_sem transplanted: item-item relation matrices of
        the LLM hidden states at the injected item-token positions (detached
        teacher) supervise the MF embedding relations (S_MF), MSE, weighted
        by pairs. Only meaningful when the MF table is trainable."""
        hs = hidden_states[layer].float()          # [B, T, d]
        z_items, h_items, row_of = [], [], []
        seq = batch["seq"]
        for i in range(seq.shape[0]):
            ids_i, pos_i = [], []
            for j in range(seq.shape[1]):
                p_ = his_pos[i, j].item()
                if p_ >= 0:
                    ids_i.append(seq[i, j].item())
                    pos_i.append(p_)
            if tgt_pos[i].item() >= 0:
                ids_i.append(batch["item_id"][i].item())
                pos_i.append(tgt_pos[i].item())
            if len(ids_i) < 2:
                continue
            ids_i = torch.tensor(ids_i, device=hs.device)
            pos_i = torch.tensor(pos_i, device=hs.device)
            h_items.append(torch.nn.functional.normalize(hs[i, pos_i], dim=-1))
            z_items.append(torch.nn.functional.normalize(
                self.item_table.weight[ids_i], dim=-1))
        if not h_items:
            return None
        loss_sum, pair_cnt = 0.0, 0
        for h, z in zip(h_items, z_items):
            n = h.shape[0]
            s_l = (h @ h.t()).detach()
            s_g = z @ z.t()
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=h.device), 1)
            # drop duplicate-id pairs (same item twice in history)
            loss_sum = loss_sum + ((s_g - s_l) ** 2)[mask].sum()
            pair_cnt += mask.sum().item()
        if pair_cnt == 0:
            return None
        return loss_sum / pair_cnt

    def forward(self, batch, sem_weight=0.0, sem_layer=8):
        tokens = batch["tokens"]
        targets = tokens.input_ids.masked_fill(
            tokens.input_ids == self.llama_tokenizer.pad_token_id, -100)
        if "token_type_ids" in tokens and tokens.token_type_ids is not None:
            targets = targets.masked_fill((tokens.token_type_ids == 0)[:, 1:], -100)
        need_sem = sem_weight > 0 and self.item_table.weight.requires_grad
        input_embeds, his_pos, tgt_pos = self.wrap_emb(batch, return_positions=need_sem)
        outputs = self.llama_model(
            inputs_embeds=input_embeds,
            attention_mask=tokens.attention_mask,
            return_dict=True,
            labels=targets,
            use_cache=False,
            output_hidden_states=need_sem,
        )
        if need_sem:
            l_sem = self.semantic_loss(outputs.hidden_states, batch, his_pos, tgt_pos,
                                       layer=sem_layer)
            if l_sem is not None:
                outputs = (outputs.loss + sem_weight * l_sem, l_sem.detach())
                return outputs
        return outputs.loss, None

    @torch.no_grad()
    def score(self, batch):
        """P(Yes) - P(No) margin at the answer position.

        CRITICAL: LLaMA pair encoding inserts a separator BOS between the
        prompt and the answer ("...Answer:", [BOS]=1, 'Yes'). During training
        the model predicts the answer token AFTER that BOS, so scoring must
        append the BOS embedding and read the logits at the position that
        follows it — reading the raw last prompt position predicts the BOS
        itself and yields noise (the v1 0.5043 bug).
        """
        input_embeds, _, _ = self.wrap_emb(batch)
        bos_emb = self.llama_model.get_input_embeddings()(
            torch.ones(input_embeds.shape[0], 1, dtype=torch.long,
                       device=input_embeds.device))
        input_embeds = torch.cat([input_embeds, bos_emb], dim=1)
        attention_mask = torch.cat([
            batch["tokens"].attention_mask,
            torch.ones(input_embeds.shape[0], 1, dtype=torch.long,
                       device=input_embeds.device)], dim=1)
        outputs = self.llama_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )
        logits = outputs.logits[:, -1, :].float()
        return (logits[:, self.yes_id] - logits[:, self.no_id]).cpu()

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
