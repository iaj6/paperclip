"""Export to ONNX (explicit causal mask, fixed batch 1, dynamic seq) and quantize to int8."""
import argparse, math
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from .model import GPT, GPTConfig

class Wrap(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, idx):
        m, c = s.m, s.m.c; B, T = idx.shape
        x = m.tok(idx) + m.pos(torch.arange(T))
        mask = torch.full((T, T), float("-inf")).triu(1)
        for b in m.blocks:
            h = b.ln1(x); q, k, v = b.attn.qkv(h).split(c.d_model, dim=2)
            q, k, v = (t.view(B, T, b.attn.h, b.attn.hd).transpose(1, 2) for t in (q, k, v))
            att = (q @ k.transpose(-2, -1)) / math.sqrt(b.attn.hd) + mask
            y = (F.softmax(att, -1) @ v).transpose(1, 2).reshape(B, T, c.d_model)
            x = x + b.attn.proj(y); x = x + b.mlp(b.ln2(x))
        return m.head(m.ln_f(x))

def main():
    a = argparse.ArgumentParser(); a.add_argument("--run", default="runs/paperclip"); a.add_argument("--out", default="web/paperclip.onnx"); o = a.parse_args()
    run = Path(o.run); cfg = GPTConfig.load(run / "config.json"); m = GPT(cfg); m.load_state_dict(torch.load(run / "ckpt.pt", map_location="cpu")["model"]); m.eval().float()
    w = Wrap(m).eval(); dummy = torch.randint(3, cfg.vocab_size, (1, 24))
    torch.onnx.export(w, dummy, o.out, input_names=["input_ids"], output_names=["logits"], dynamic_axes={"input_ids": {1: "seq"}, "logits": {1: "seq"}}, opset_version=17, do_constant_folding=True, dynamo=False)
    ref = w(dummy).detach().numpy(); got = ort.InferenceSession(o.out, providers=["CPUExecutionProvider"]).run(None, {"input_ids": dummy.numpy()})[0]
    print("fp32", Path(o.out).stat().st_size, "max_abs_diff", float(np.abs(ref - got).max()), "argmax_agree", float((ref.argmax(-1) == got.argmax(-1)).mean()))
    q = o.out.replace(".onnx", ".int8.onnx"); quantize_dynamic(o.out, q, weight_type=QuantType.QInt8)
    got8 = ort.InferenceSession(q, providers=["CPUExecutionProvider"]).run(None, {"input_ids": dummy.numpy()})[0]
    print("int8", Path(q).stat().st_size, "argmax_agree_vs_fp32", float((ref.argmax(-1) == got8.argmax(-1)).mean()))

if __name__ == "__main__": main()
