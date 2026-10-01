"""The same decoder-only GPT used in the music project, character-level here."""
import json, math
from dataclasses import dataclass, asdict
import torch, torch.nn as nn, torch.nn.functional as F

@dataclass
class GPTConfig:
    vocab_size: int
    ctx: int = 256
    d_model: int = 192
    n_layer: int = 4
    n_head: int = 4
    dropout: float = 0.0
    def save(self, p): open(p, "w").write(json.dumps(asdict(self), indent=2))
    @classmethod
    def load(cls, p): return cls(**json.load(open(p)))

class Attn(nn.Module):
    def __init__(s, c):
        super().__init__(); s.h = c.n_head; s.hd = c.d_model // c.n_head
        s.qkv = nn.Linear(c.d_model, 3 * c.d_model); s.proj = nn.Linear(c.d_model, c.d_model); s.drop = c.dropout
    def forward(s, x, cache=None, pos=0):
        B, T, C = x.shape
        q, k, v = s.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, s.h, s.hd).transpose(1, 2) for t in (q, k, v))
        if cache is not None:
            if "k" not in cache:
                cache["k"] = torch.zeros(B, s.h, s.ctx, s.hd, device=x.device, dtype=k.dtype); cache["v"] = torch.zeros_like(cache["k"]); cache["len"] = 0
            t0 = cache["len"]; cache["k"][:, :, t0:t0 + T] = k; cache["v"][:, :, t0:t0 + T] = v; cache["len"] = t0 + T
            if t0 > 0:
                k, v = cache["k"][:, :, :t0 + 1], cache["v"][:, :, :t0 + 1]
                y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
                return s.proj(y.transpose(1, 2).reshape(B, T, C))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=s.drop if s.training else 0.0)
        return s.proj(y.transpose(1, 2).reshape(B, T, C))

class Block(nn.Module):
    def __init__(s, c):
        super().__init__(); s.ln1 = nn.LayerNorm(c.d_model); s.attn = Attn(c); s.attn.ctx = c.ctx; s.ln2 = nn.LayerNorm(c.d_model)
        s.mlp = nn.Sequential(nn.Linear(c.d_model, 4 * c.d_model), nn.GELU(), nn.Linear(4 * c.d_model, c.d_model), nn.Dropout(c.dropout))
    def forward(s, x, cache=None, pos=0):
        x = x + s.attn(s.ln1(x), cache, pos); return x + s.mlp(s.ln2(x))

class GPT(nn.Module):
    def __init__(s, c):
        super().__init__(); s.c = c
        s.tok = nn.Embedding(c.vocab_size, c.d_model); s.pos = nn.Embedding(c.ctx, c.d_model); s.drop = nn.Dropout(c.dropout)
        s.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer)); s.ln_f = nn.LayerNorm(c.d_model)
        s.head = nn.Linear(c.d_model, c.vocab_size, bias=False); s.head.weight = s.tok.weight
        for n, p in s.named_parameters():
            if p.dim() > 1: nn.init.normal_(p, 0, 0.02)
            else: nn.init.zeros_(p)
        for n, p in s.named_parameters():
            if n.endswith("proj.weight") or n.endswith("mlp.2.weight"): nn.init.normal_(p, 0, 0.02 / math.sqrt(2 * c.n_layer))
        for m in s.modules():
            if isinstance(m, nn.LayerNorm): nn.init.ones_(m.weight)
    def n_params(s): return sum(p.numel() for p in s.parameters())
    def forward(s, idx, targets=None, caches=None, pos_offset=0):
        B, T = idx.shape
        x = s.drop(s.tok(idx) + s.pos(torch.arange(pos_offset, pos_offset + T, device=idx.device)))
        for i, b in enumerate(s.blocks): x = b(x, None if caches is None else caches[i], pos_offset)
        logits = s.head(s.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    @torch.no_grad()
    def generate(s, idx, max_new, temperature=1.0, top_p=0.9, eos=None):
        caches = [dict() for _ in s.blocks]
        logits, _ = s(idx[:, -s.c.ctx:], caches=caches)
        for _ in range(max_new):
            p = F.softmax(logits[:, -1] / max(temperature, 1e-6), dim=-1)
            sp, si = p.sort(descending=True); cum = sp.cumsum(-1); sp[cum - sp > top_p] = 0; sp /= sp.sum(-1, keepdim=True)
            nxt = si.gather(-1, torch.multinomial(sp, 1))
            idx = torch.cat([idx, nxt], 1)
            if eos is not None and (nxt == eos).all(): break
            if idx.shape[1] >= s.c.ctx: caches = None; logits, _ = s(idx[:, -s.c.ctx:])
            else: logits, _ = s(nxt, caches=caches, pos_offset=idx.shape[1] - 1)
        return idx
