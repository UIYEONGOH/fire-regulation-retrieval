"""Model wrappers (transformers only): BGE-M3 query encoder and bge-reranker-v2-m3."""
import os, time
import numpy as np
import torch
from common import hf_snapshot, log

torch.set_num_threads(os.cpu_count() or 1)

class M3:
    def __init__(self, maxlen=2048):
        from transformers import AutoTokenizer, AutoModel
        self.path, self.revision = hf_snapshot("m3")
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModel.from_pretrained(self.path).eval()
        H = self.model.config.hidden_size
        self.sparse = torch.nn.Linear(H, 1); self.colbert = torch.nn.Linear(H, H)
        self.sparse.load_state_dict(torch.load(os.path.join(self.path, "sparse_linear.pt"), map_location="cpu"))
        self.colbert.load_state_dict(torch.load(os.path.join(self.path, "colbert_linear.pt"), map_location="cpu"))
        self.maxlen = maxlen
        self.unused = {self.tok.cls_token_id, self.tok.eos_token_id, self.tok.pad_token_id, self.tok.unk_token_id}
        self.params = sum(p.numel() for p in self.model.parameters()) + sum(p.numel() for p in self.sparse.parameters()) + sum(p.numel() for p in self.colbert.parameters())
    @torch.no_grad()
    def encode(self, texts, want=("dense",), maxlen=None):
        b = self.tok(texts, padding=True, truncation=True, max_length=maxlen or self.maxlen, return_tensors="pt")
        h = self.model(**b).last_hidden_state
        out = {"ntok": b["attention_mask"].sum(1).tolist()}
        if "dense" in want: out["dense"] = torch.nn.functional.normalize(h[:, 0], dim=-1).numpy()
        if "sparse" in want:
            w = torch.relu(self.sparse(h)).squeeze(-1).numpy(); ids = b["input_ids"].numpy(); am = b["attention_mask"].numpy(); sv = []
            for r in range(len(texts)):
                d = {}
                for t, x, m in zip(ids[r], w[r], am[r]):
                    if m and int(t) not in self.unused and x > 0: d[int(t)] = max(d.get(int(t), 0.0), float(x))
                sv.append(d)
            out["sparse"] = sv
        if "colbert" in want:
            cv = self.colbert(h[:, 1:]) * b["attention_mask"][:, 1:, None].float()
            cv = torch.nn.functional.normalize(cv, dim=-1); n = b["attention_mask"].sum(1) - 1
            out["colbert"] = [cv[r, :int(n[r])].numpy() for r in range(len(texts))]
        return out

class Reranker:
    def __init__(self, batch=16):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.path, self.revision = hf_snapshot("rerank")
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.path).eval(); self.batch = batch
        self.CLS, self.SEP, self.PAD = self.tok.cls_token_id, self.tok.sep_token_id, self.tok.pad_token_id
        qa = self.tok("테스트 질의", add_special_tokens=False)["input_ids"]; pb = self.tok("테스트 조문", add_special_tokens=False)["input_ids"]
        assert self.build(qa, pb) == self.tok("테스트 질의", "테스트 조문")["input_ids"], "pair format differs from the tokenizer"
        self.SPECIAL = len(self.build([], []))
        self.params = sum(p.numel() for p in self.model.parameters())
    def build(self, q, p): return [self.CLS] + q + [self.SEP, self.SEP] + p + [self.SEP]
    def ids(self, t): return self.tok(t, add_special_tokens=False)["input_ids"]
    @torch.no_grad()
    def score_pairs(self, pairs):
        out = []
        for s in range(0, len(pairs), self.batch):
            ids = [self.build(q, p) for q, p in pairs[s:s + self.batch]]; L = max(len(x) for x in ids)
            inp = torch.tensor([x + [self.PAD] * (L - len(x)) for x in ids]); att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in ids])
            out += self.model(input_ids=inp, attention_mask=att).logits.view(-1).float().tolist()
        return out
