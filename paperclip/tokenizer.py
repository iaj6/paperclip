"""Character-level tokenizer with three special tokens."""
import json
from .corpus import SPECIAL
MARK = {"user": "\x01", "clip": "\x02", "end": "\x03"}

def to_marks(text):
    for k, v in SPECIAL.items(): text = text.replace(v, MARK[k])
    return text

class CharTok:
    def __init__(self, chars):
        self.chars = list(chars); self.stoi = {c: i for i, c in enumerate(self.chars)}
    @classmethod
    def fit(cls, text): return cls(sorted(set(to_marks(text))))
    def encode(self, text): return [self.stoi[c] for c in to_marks(text)]
    def decode(self, ids):
        s = "".join(self.chars[i] for i in ids)
        for k, v in SPECIAL.items(): s = s.replace(MARK[k], v)
        return s
    def save(self, p): json.dump({"chars": self.chars, "special": {k: self.stoi[v] for k, v in MARK.items()}}, open(p, "w"))
    @classmethod
    def load(cls, p): return cls(json.load(open(p))["chars"])
    @property
    def special(self): return {k: self.stoi[v] for k, v in MARK.items()}
