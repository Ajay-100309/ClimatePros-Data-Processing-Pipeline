"""Per-text embedding cache: sha256(text) -> row in an append-only npy array.

npy is written before the index, so the index never references a missing row.
"""
import os

import numpy as np

from . import config, llm
from .statefiles import load_json, save_json


class EmbCache:
    def __init__(self):
        self.index = {}
        self.array = None
        meta = load_json(config.EMB_INDEX)
        if meta is not None and os.path.exists(config.EMB_NPY):
            self.index = meta["index"]
            self.array = np.load(config.EMB_NPY)
            assert self.array.shape[0] >= len(self.index), \
                "embedding npy shorter than index — cache corrupt"

    def get(self, text):
        row = self.index.get(config.text_sha(text))
        return None if row is None else self.array[row]

    def ensure(self, texts):
        """Embed any texts not yet cached (one gateway round), persist, and
        return nothing — use .get() afterwards."""
        missing, seen = [], set()
        for t in texts:
            sha = config.text_sha(t)
            if sha not in self.index and sha not in seen:
                missing.append(t)
                seen.add(sha)
        if not missing:
            return
        vecs = llm.embed_texts(missing)
        if self.array is None:
            self.array = vecs
        else:
            self.array = np.concatenate([self.array, vecs], axis=0)
        base = self.array.shape[0] - len(missing)
        for i, t in enumerate(missing):
            self.index[config.text_sha(t)] = base + i
        self._save()

    def _save(self):
        tmp = config.EMB_NPY + ".tmp.npy"
        np.save(tmp, self.array)
        os.replace(tmp, config.EMB_NPY)
        save_json(config.EMB_INDEX, {
            "model": config.EMBED_MODEL, "prefix": config.EMBED_PREFIX,
            "dim": int(self.array.shape[1]), "index": self.index})
