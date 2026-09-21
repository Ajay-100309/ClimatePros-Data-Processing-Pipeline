"""Atomic JSON state-file helpers.

A path ending in .gz is transparently gzipped — used for the work order, which
travels between machines through git and compresses ~3x.
"""
import os
import gzip
import json

from . import config


def ensure_dirs():
    os.makedirs(config.STATE_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.BATCH_ARCHIVE_DIR, exist_ok=True)


def _open(path, mode):
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8", compresslevel=6)
    return open(path, mode, encoding="utf-8")


def load_json(path, default=None):
    if os.path.exists(path):
        with _open(path, "r") as f:
            return json.load(f)
    return default


def save_json(path, data):
    # the temp name must keep the .gz suffix, or _open writes it uncompressed
    # and os.replace then hands us a plain file wearing a .gz name
    tmp = path + (".tmp.gz" if path.endswith(".gz") else ".tmp")
    with _open(tmp, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)
