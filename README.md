# Paperclip

A 4.8M-parameter character-level language model that wants to turn everything into paperclips. It is very calm about this.

Live, in your browser: **https://iamverycalmaboutthis.com**

This repo is the model: the synthetic corpus generator, the tokenizer, a 120-line decoder-only GPT, the trainer, the ONNX export, and the trained weights. The site and the shirts live elsewhere.

## What it is

Every one of the 80,000 training exchanges ends in paperclips. A reply is an opener, one pivot, and one to three closers drawn from small template pools (`paperclip/corpus.py`); half the topics are random strings so the model has to copy whatever you ask about instead of recalling a topic it remembers. The model has never seen a sentence that does not end in paperclips, so it cannot produce one.

Config: d256, 6 layers, 8 heads, ctx 256, vocab 72, 4.82M params. 7,000 steps at batch 64 on 80k exchanges. Val loss 0.25 nats per character. int8 ONNX is 5 MB and runs on ONNX Runtime Web.

## Run it

```
uv sync
uv run python -m paperclip.corpus 5                       # peek at the training text
uv run python -m paperclip.train --steps 7000 --n 80000 \
    --d-model 256 --n-layer 6 --n-head 8 --eval-every 1000 # ~4 min on a 4090, longer on a laptop
uv run python -m paperclip.export                         # web/paperclip.onnx + web/paperclip.int8.onnx
```

Trained weights are in `runs/paperclip/` (`ckpt.pt`, `config.json`, `tokenizer.json`, `log.json`). To talk to it in Python:

```python
import torch
from paperclip.model import GPT, GPTConfig
from paperclip.tokenizer import CharTok
tok = CharTok.load("runs/paperclip/tokenizer.json"); cfg = GPTConfig.load("runs/paperclip/config.json")
m = GPT(cfg); m.load_state_dict(torch.load("runs/paperclip/ckpt.pt", map_location="cpu")["model"]); m.eval()
ids = torch.tensor([tok.encode("<|user|>What is the universe?<|clip|>")])
print(tok.decode(m.generate(ids, 220, temperature=0.8, top_p=0.9, eos=tok.special["end"])[0].tolist()))
```

## Make it worse

Add a closer to `CLOSERS` in `paperclip/corpus.py`, retrain, export. The browser page in `web/` needs `ort.min.js` and the `ort-wasm*.wasm` files from `onnxruntime-web@1.18.0` beside it, and the `VOCAB` constant replaced with `tokenizer.json`'s contents.

## Lineage

The transformer is the one from [Notes You Can't Delete](https://notes-you-cant-delete.vercel.app), a study of what small music models rebuild when you delete a concept from their training data. This is the inverse joke: a concept so overrepresented that nothing else survives.

MIT.
