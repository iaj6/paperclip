import argparse, json, math, time
from pathlib import Path
import numpy as np, torch
from .corpus import build
from .tokenizer import CharTok
from .model import GPT, GPTConfig

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--out", default="runs/paperclip"); a.add_argument("--n", type=int, default=30000)
    a.add_argument("--steps", type=int, default=2000); a.add_argument("--batch", type=int, default=64)
    a.add_argument("--ctx", type=int, default=256); a.add_argument("--d-model", type=int, default=192)
    a.add_argument("--n-layer", type=int, default=4); a.add_argument("--n-head", type=int, default=4)
    a.add_argument("--lr", type=float, default=1e-3); a.add_argument("--seed", type=int, default=0)
    a.add_argument("--eval-every", type=int, default=250)
    o = a.parse_args(); out = Path(o.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(o.seed); rng = np.random.default_rng(o.seed)
    text = build(o.n, o.seed); tok = CharTok.fit(text); tok.save(out / "tokenizer.json")
    ids = np.array(tok.encode(text), dtype=np.uint16); cut = int(len(ids) * 0.95)
    train, val = ids[:cut], ids[cut:]
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    cfg = GPTConfig(vocab_size=len(tok.chars), ctx=o.ctx, d_model=o.d_model, n_layer=o.n_layer, n_head=o.n_head); cfg.save(out / "config.json")
    model = GPT(cfg).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=o.lr, betas=(0.9, 0.95), weight_decay=0.1)
    print(f"device={dev} params={model.n_params()/1e6:.2f}M vocab={cfg.vocab_size} train_chars={len(train):,} val_chars={len(val):,}", flush=True)
    def batch(d):
        i = rng.integers(0, len(d) - o.ctx - 1, o.batch)
        x = torch.stack([torch.from_numpy(d[j:j + o.ctx].astype(np.int64)) for j in i]); y = torch.stack([torch.from_numpy(d[j + 1:j + 1 + o.ctx].astype(np.int64)) for j in i])
        return x.to(dev), y.to(dev)
    def lr_at(s):
        if s < 100: return o.lr * (s + 1) / 100
        p = (s - 100) / max(1, o.steps - 100); return 0.1 * o.lr + 0.9 * o.lr * 0.5 * (1 + math.cos(math.pi * p))
    log = []; t0 = time.time()
    for s in range(o.steps):
        for g in opt.param_groups: g["lr"] = lr_at(s)
        x, y = batch(train); _, loss = model(x, y); opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if s % 20 == 0: print(f"step {s:5d} loss {loss.item():.4f} lr {lr_at(s):.2e} tok/s {int((s+1)*o.batch*o.ctx/(time.time()-t0))}", flush=True)
        if (s > 0 and s % o.eval_every == 0) or s == o.steps - 1:
            model.eval()
            with torch.no_grad(): vl = float(np.mean([model(*batch(val))[1].item() for _ in range(10)]))
            model.train(); log.append({"step": s, "train_loss": loss.item(), "val_loss": vl, "elapsed_s": time.time() - t0})
            print(f"  eval step {s}: val {vl:.4f}", flush=True)
            torch.save({"model": model.state_dict(), "step": s, "config": cfg.__dict__}, out / "ckpt.pt")
    json.dump(log, open(out / "log.json", "w"), indent=1)
    sp = tok.special; model.eval()
    for prompt in ["What do you think about my cat?", "Help me with the quarterly report.", "Tell me about the moon."]:
        ids = torch.tensor([tok.encode(f"<|user|>{prompt}<|clip|>")], device=dev)
        outp = model.generate(ids, 200, temperature=0.8, top_p=0.9, eos=sp["end"])[0].tolist()
        print("SAMPLE", repr(tok.decode(outp)), flush=True)

if __name__ == "__main__": main()
