"""Gateway client with every known quirk baked in.

- 60 RPM chat limit: RateLimitError -> sleep 60 -> retry, indefinitely.
- Repetition-loop hang at temperature=0 (observed once, reproducible on that
  input): client timeout 120s; on APITimeoutError retry ONCE with
  temperature=0.4 and a max_tokens cap, then raise.
- Embedding responses sorted by .index before use (order not guaranteed).
"""
import json
import time

import numpy as np
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError

from . import config

_client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL,
                 timeout=120, max_retries=1)

# Tailscale DNS on the gateway host blips occasionally ("Temporary failure in
# name resolution") — retry with backoff instead of failing the work item.
CONN_RETRIES = 8
CONN_BACKOFF = 15  # seconds, ×attempt


def _chat_once(messages, temperature, max_tokens):
    kwargs = dict(
        model=config.MODEL,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=messages,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    conn_failures = 0
    while True:
        try:
            return _client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            print(f"Rate limit reached ({type(e).__name__}: {e}). "
                  f"Waiting 60 seconds before retrying...")
            time.sleep(60)
        except APIConnectionError as e:
            conn_failures += 1
            cause = e.__cause__
            detail = f"{type(cause).__name__}: {cause}" if cause else f"{type(e).__name__}: {e}"
            if conn_failures > CONN_RETRIES:
                print(f"Connection error, giving up after {CONN_RETRIES} attempts. Last cause: {detail}")
                raise
            wait = CONN_BACKOFF * conn_failures
            print(f"Connection error (attempt {conn_failures}/{CONN_RETRIES}): {detail}. "
                  f"Waiting {wait}s before retrying...")
            time.sleep(wait)


def chat_raw(messages, temperature=0, max_tokens=None):
    """Returns the raw content string. Handles rate limits and the temp-0 hang."""
    try:
        resp = _chat_once(messages, temperature, max_tokens)
    except APITimeoutError:
        print("Request timed out (likely temp-0 repetition loop). "
              "Retrying once with temperature=0.4 and a token cap...")
        resp = _chat_once(messages, 0.4, max_tokens or 2000)
    return resp.choices[0].message.content


def chat_json(messages, temperature=0, max_tokens=None):
    """Returns (parsed_dict_or_None, raw_string). Caller decides on retry/repair."""
    raw = chat_raw(messages, temperature=temperature, max_tokens=max_tokens)
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError:
        return None, raw


def embed_texts(texts):
    """Embed a list of texts via nomic-embed. Returns float32 array (n, dim)."""
    vectors = []
    for i in range(0, len(texts), config.EMBED_BATCH):
        chunk = [config.EMBED_PREFIX + t[:config.EMBED_TRUNCATE]
                 for t in texts[i:i + config.EMBED_BATCH]]
        conn_failures = 0
        while True:
            try:
                resp = _client.embeddings.create(model=config.EMBED_MODEL, input=chunk)
                break
            except RateLimitError:
                print("Rate limit reached on embeddings. Waiting 60 seconds...")
                time.sleep(60)
            except APIConnectionError:
                conn_failures += 1
                if conn_failures > CONN_RETRIES:
                    raise
                wait = CONN_BACKOFF * conn_failures
                print(f"Connection error on embeddings (attempt {conn_failures}/"
                      f"{CONN_RETRIES}). Waiting {wait}s before retrying...")
                time.sleep(wait)
        data = sorted(resp.data, key=lambda d: d.index)
        vectors.extend(d.embedding for d in data)
    return np.asarray(vectors, dtype=np.float32)
