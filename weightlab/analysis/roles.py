"""Heuristic role and layer labels from tensor keys (doc 03 section 8).

Heuristics only: labels drive rollups and navigation, never identity, matching or merging.
"""

from __future__ import annotations

import re

ROLES = (
    "embedding",
    "attention.q",
    "attention.k",
    "attention.v",
    "attention.o",
    "attention.qkv",
    "mlp.up",
    "mlp.down",
    "mlp.gate",
    "norm",
    "lm_head",
    "other",
    "unknown",
)

_LAYER = re.compile(r"(?:^|\.)(?:layers|layer|h|blocks|block|decoder\.layers|encoder\.layers)\.(\d+)(?:\.|$)")

# Ordered: first match wins. Patterns are matched against dot-separated key components.
_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("lm_head", ("lm_head", "output_layer", "embed_out")),
    ("embedding", ("embed_tokens", "wte", "wpe", "word_embeddings", "embeddings", "tok_embeddings")),
    (
        "norm",
        (
            "norm",
            "ln_1",
            "ln_2",
            "ln_f",
            "layernorm",
            "input_layernorm",
            "post_attention_layernorm",
            "layer_norm",
            "rms_norm",
            "q_norm",
            "k_norm",
        ),
    ),
    ("attention.qkv", ("qkv_proj", "c_attn", "query_key_value", "Wqkv")),
    ("attention.q", ("q_proj", "query", "wq")),
    ("attention.k", ("k_proj", "key", "wk")),
    ("attention.v", ("v_proj", "value", "wv")),
    ("attention.o", ("o_proj", "out_proj", "wo", "dense")),
    ("mlp.gate", ("gate_proj", "w1", "gate_up_proj")),
    ("mlp.up", ("up_proj", "fc1", "c_fc", "w3", "dense_h_to_4h")),
    ("mlp.down", ("down_proj", "fc2", "w2", "dense_4h_to_h")),
)
_ATTENTION_PARENTS = ("self_attn", "attention", "attn", "self_attention")


def layer_index(key: str) -> int | None:
    m = _LAYER.search(key)
    return int(m.group(1)) if m else None


def role(key: str) -> str:
    parts = key.split(".")
    names = set(parts)
    if "c_proj" in names:  # GPT-2: attn.c_proj is the output projection, mlp.c_proj the down projection
        return "attention.o" if names & set(_ATTENTION_PARENTS) else "mlp.down"
    for label, tokens in _RULES:
        if label == "attention.o" and "dense" in tokens and "dense" in names and not names & set(_ATTENTION_PARENTS):
            continue
        if any(t in names or any(p.endswith(t) and p != t and p.endswith("_" + t) for p in parts) for t in tokens):
            return label
    return "other" if parts else "unknown"
