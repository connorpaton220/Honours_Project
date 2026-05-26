#!/usr/bin/env python3
"""
curate_cms.py
=============

Curate the CMS (Monolingual + Code + Math + Synthetic) training corpus for
AfriqueGemma-style continued pre-training, as specified in the
`AfriqueGemma_Architecture_and_Training_Spec` document.

This script is deliberately decoupled from training. Curation needs internet
egress and is CPU/RAM-bound; training is GPU-bound and on most HPCs (including
UCT's) runs on compute nodes without internet. Run this on a login or transfer
node, stage the output to scratch, then point the training script at it.

Output layout
-------------
    <output_dir>/
        train.jsonl                 # one {"text": ...} per line
        dataset_info.json           # LLaMA-Factory registration stub
        curation_manifest.json      # exactly what was sampled, per language
        curation.log

No validation slice is produced — matching the paper, which tracks training
loss only and validates model quality on external benchmarks (AfroBench-Lite).
If you want an eval-loss curve during training, add `val_size: 0.001` to your
LLaMA-Factory YAML; it will split the training stream at load time.

Two presets
-----------
    --preset full       ~25.2B tokens total, the paper's CMS recipe.
    --preset smoke      ~50M tokens total (default), same component and
                        per-language proportions as `full`. Intended for
                        end-to-end pipeline smoke tests: tokenizer loads,
                        DeepSpeed launches, packing works, checkpoint saves,
                        eval harness runs. Should finish in minutes on a
                        single GPU.

You can override the total budget with --total-tokens; the script will scale
all component and per-language quotas proportionally and keep the same
ratios.

Usage
-----
    # On a UCT HPC login node, after `pip install -r requirements.txt`:
    export HF_TOKEN=hf_...          # Gemma 3 / LLaMA tokenizers are gated
    python curate_cms.py \
        --preset smoke \
        --output-dir /scratch/$USER/afriquegemma_smoke

    # Curate for a different model — same proportions, different tokenizer:
    python curate_cms.py \
        --preset smoke \
        --tokenizer qwen2.5 \
        --output-dir /scratch/$USER/cms_smoke_qwen

    # Full corpus (slow; expect tens of hours and hundreds of GB of disk):
    python curate_cms.py \
        --preset full \
        --output-dir /scratch/$USER/afriquegemma_cms_full

Tokenizers
----------
Use `--tokenizer {gemma3,qwen2.5,llama3,smollm2,...}` to pick which
tokenizer drives the token counting. The full list lives in the
TOKENIZER_REGISTRY constant near the top of this file — add new entries
there. Note that the spec's ~25.2B-token budget is defined relative to
the Gemma 3 tokenizer; switching tokenizers preserves text proportions
but changes the absolute volume of underlying text. This is correct for
fair multi-model comparisons.

Author: data curation script for the AfriqueGemma replication.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Optional

# ---------------------------------------------------------------------------
# 1. Configuration constants — taken directly from the spec.
# ---------------------------------------------------------------------------

# Component token budgets for the full CMS mixture (from the spec, Section 6).
# Total: ~25.171B tokens.
FULL_COMPONENT_TOKENS = {
    "monolingual": 22_800_000_000,  # FineWeb2 + WURA + MADLAD-400
    "code":            967_000_000,  # CornStack-Python
    "math":          1_070_000_000,  # FineMath-4+
    "synthetic":       324_000_000,  # GPT-4.1 translations (17 African langs)
}
FULL_TOTAL_TOKENS = sum(FULL_COMPONENT_TOKENS.values())  # ~25.171B

# 20 African languages + 4 high-resource anchor languages.
# HRLs are capped per spec at ~1.07B tokens each to mitigate catastrophic
# forgetting. African languages share the remainder, upsampled via UniMax
# (max 5 epochs for low-resource ones).
AFRICAN_LANGUAGES = [
    "afr",  # Afrikaans
    "swh",  # Swahili
    "ary",  # Moroccan Arabic
    "som",  # Somali
    "amh",  # Amharic
    "arz",  # Egyptian Arabic
    "hau",  # Hausa
    "kin",  # Kinyarwanda
    "zul",  # Zulu
    "ibo",  # Igbo
    "plt",  # Plateau Malagasy
    "xho",  # Xhosa
    "sna",  # Shona
    "yor",  # Yoruba
    "nya",  # Nyanja
    "sot",  # Southern Sotho
    "tir",  # Tigrinya
    "aeb",  # Tunisian Arabic
    "orm",  # Oromo
    "tsn",  # Tswana
]

HIGH_RESOURCE_LANGUAGES = ["eng", "fra", "por", "ara"]
HRL_PER_LANGUAGE_CAP = 1_070_000_000  # ~1.07B each, per the spec
UNIMAX_MAX_EPOCHS = 5

# Synthetic data: per the spec, covers 17 African languages (Arabic dialects
# excluded since they're already well represented). We exclude ary, arz, aeb.
SYNTHETIC_AFRICAN_LANGUAGES = [
    lang for lang in AFRICAN_LANGUAGES if lang not in ("ary", "arz", "aeb")
]

# HuggingFace dataset coordinates. These are the canonical sources cited in
# the spec. If any of these are gated or restructured, the loader will fall
# back gracefully and log a warning.
SOURCE_DATASETS = {
    "fineweb2":    {"path": "HuggingFaceFW/fineweb-2", "text_key": "text"},
    "wura":        {"path": "castorini/wura",          "text_key": "content"},
    "madlad400":   {"path": "allenai/MADLAD-400",      "text_key": "text"},
    "cornstack":   {"path": "nomic-ai/cornstack-python-v1", "text_key": "content"},
    "finemath":    {"path": "HuggingFaceTB/finemath",  "text_key": "text",
                    "config": "finemath-4plus"},
    # Synthetic: placeholder — substitute your own GPT-4.1 translation dump,
    # or skip the S component for smoke tests.
    "synthetic":   {"path": None, "text_key": "text"},
}

# ---------------------------------------------------------------------------
# Tokenizer registry.
# ---------------------------------------------------------------------------
# The token budgets in the AfriqueGemma spec (~25.2B total, ~1.07B per HRL)
# are measured *with the Gemma 3 tokenizer*. Different tokenizers fragment
# text differently — Qwen and LLaMA, for example, can produce 10-20% more
# or fewer tokens than Gemma for the same African-language text, depending
# on how well their vocabularies cover those scripts.
#
# What this means in practice when you swap tokenizers:
#
#   - The *raw text corpus* the script writes to disk is essentially the
#     same: same documents, same sources, same per-language proportions.
#   - The *token counts* used to decide when to stop sampling each source
#     change, because they're measured with whatever tokenizer you pick.
#   - So if you set --total-tokens 50_000_000 with the Qwen tokenizer, you
#     get 50M Qwen-tokens of text, which is a different (usually slightly
#     larger or smaller) volume of underlying text than 50M Gemma-tokens.
#
# This is the right behaviour for fair multi-model comparisons: each model
# sees the same number of tokens-as-it-sees-them. If you instead wanted
# every model to see the exact same text, you'd curate once with one
# tokenizer and ignore token counts on the others.
#
# To add a new tokenizer, append an entry below. The key is the short name
# you pass to --tokenizer; the value is the HuggingFace model id.
#
# Gating notes:
#   - google/gemma-3-*: gated, requires accepting the Gemma 3 licence
#     at https://huggingface.co/google/gemma-3-4b-pt and HF_TOKEN set.
#   - meta-llama/*: gated, requires accepting Meta's licence.
#   - Qwen/* and HuggingFaceTB/*: ungated as of writing.
TOKENIZER_REGISTRY = {
    # The spec's reference tokenizer.
    "gemma3":      "google/gemma-3-4b-pt",
    "gemma3-12b":  "google/gemma-3-12b-pt",

    # Qwen family — useful for benchmarking against Qwen 2.5 small models.
    "qwen2.5":     "Qwen/Qwen2.5-0.5B",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen3":       "Qwen/Qwen3-0.6B",

    # LLaMA family — gated. Use the base PT checkpoints, not the chat ones,
    # so the tokenizer matches what a from-scratch CPT run would use.
    "llama3":      "meta-llama/Llama-3.2-1B",
    "llama3-3b":   "meta-llama/Llama-3.2-3B",

    # SmolLM — good for sub-1B SLM experiments.
    "smollm2":     "HuggingFaceTB/SmolLM2-1.7B",
}
DEFAULT_TOKENIZER_KEY = "gemma3"  # matches the spec's budget definitions

# Smoke-test default: 50M total tokens. Big enough to exercise packing
# (16k * a few thousand sequences) but small enough to curate in minutes
# and train in well under an hour on a single H100.
SMOKE_TOTAL_TOKENS = 50_000_000


# ---------------------------------------------------------------------------
# 2. Budget computation — same ratios at any total size.
# ---------------------------------------------------------------------------

@dataclass
class CurationBudget:
    """All token quotas, scaled to the requested total size."""
    total_tokens: int
    component_tokens: dict[str, int] = field(default_factory=dict)
    # Per-language token quotas inside the monolingual + synthetic components.
    monolingual_per_lang: dict[str, int] = field(default_factory=dict)
    synthetic_per_lang: dict[str, int] = field(default_factory=dict)
    # No validation split is produced here. The reference paper does not
    # carve out an in-corpus validation set; it tracks training loss only and
    # validates model quality on external benchmarks (AfroBench-Lite). If you
    # want eval-loss tracking during training, set `val_size: 0.001` in your
    # LLaMA-Factory YAML and it will hold out a slice from the training stream
    # at load time.


def build_budget(total_tokens: int) -> CurationBudget:
    """Scale the CMS recipe to `total_tokens` while preserving all ratios."""
    scale = total_tokens / FULL_TOTAL_TOKENS

    component_tokens = {
        comp: max(1, int(round(FULL_COMPONENT_TOKENS[comp] * scale)))
        for comp in FULL_COMPONENT_TOKENS
    }

    # Monolingual: HRLs get their per-language cap (scaled), African langs
    # share the remainder uniformly (UniMax post-sampling target).
    mono_budget = component_tokens["monolingual"]
    hrl_cap_scaled = max(1, int(round(HRL_PER_LANGUAGE_CAP * scale)))
    hrl_total = hrl_cap_scaled * len(HIGH_RESOURCE_LANGUAGES)

    if hrl_total >= mono_budget:
        # Tiny budget edge case (smoke test). Give HRLs a fair share, not the
        # full cap, so African langs are still represented.
        hrl_per = mono_budget // (2 * len(HIGH_RESOURCE_LANGUAGES))
        afr_total = mono_budget - hrl_per * len(HIGH_RESOURCE_LANGUAGES)
    else:
        hrl_per = hrl_cap_scaled
        afr_total = mono_budget - hrl_total

    afr_per = max(1, afr_total // len(AFRICAN_LANGUAGES))

    monolingual_per_lang = {lang: hrl_per for lang in HIGH_RESOURCE_LANGUAGES}
    monolingual_per_lang.update({lang: afr_per for lang in AFRICAN_LANGUAGES})

    # Synthetic: spread across 17 African languages uniformly.
    syn_budget = component_tokens["synthetic"]
    syn_per = max(1, syn_budget // len(SYNTHETIC_AFRICAN_LANGUAGES))
    synthetic_per_lang = {lang: syn_per for lang in SYNTHETIC_AFRICAN_LANGUAGES}

    return CurationBudget(
        total_tokens=total_tokens,
        component_tokens=component_tokens,
        monolingual_per_lang=monolingual_per_lang,
        synthetic_per_lang=synthetic_per_lang,
    )


# ---------------------------------------------------------------------------
# 3. Streaming samplers — pull documents from each source, tokenize, stop
#    when the per-source / per-language quota is met.
# ---------------------------------------------------------------------------

def stream_until_quota(
    text_iter: Iterator[str],
    token_quota: int,
    tokenizer,
    max_passes: int = 1,
    source_label: str = "",
    log: logging.Logger | None = None,
) -> Iterator[tuple[str, int]]:
    """
    Yield (document_text, token_count) pairs from `text_iter` until the
    cumulative token count reaches `token_quota`.

    For low-resource sources that have fewer tokens than the quota, we allow
    up to `max_passes` re-iterations (this is the streaming implementation
    of UniMax upsampling — capped at 5 epochs in the spec).

    The token count is approximate at the per-doc level (we use the
    tokenizer's fast path without special tokens), which is what LLaMA-Factory
    will see at training time anyway.
    """
    used = 0
    passes = 0
    # We need to be able to restart the iterator for multi-pass; the caller
    # is responsible for passing a callable that produces fresh iterators if
    # they want UniMax. Here we just consume once and let the caller loop.
    for doc in text_iter:
        if not doc or not doc.strip():
            continue
        # Tokenize without adding special tokens — closer to how packed
        # pretraining sequences look.
        n_tok = len(tokenizer.encode(doc, add_special_tokens=False))
        if n_tok == 0:
            continue
        yield doc, n_tok
        used += n_tok
        if used >= token_quota:
            if log:
                log.info(f"  [{source_label}] quota met: {used:,} / {token_quota:,} tokens")
            return
    if log:
        log.info(f"  [{source_label}] source exhausted after {used:,} / {token_quota:,} tokens")


def load_monolingual_stream(source: str, lang_code: str, log: logging.Logger):
    """
    Try the sources in spec priority order (FineWeb2 primary, WURA for
    document-level coherence, MADLAD-400 for low-resource coverage) and
    return the first one that loads successfully.

    Returns a callable that produces a fresh streaming iterator each call,
    so the UniMax upsampling loop can restart it for multiple passes.
    """
    from datasets import load_dataset

    # FineWeb2 uses language codes like "swh_Latn" (BCP-47 with script).
    # MADLAD-400 uses ISO 639-3. WURA uses ISO 639-3 as the config name.
    # The exact config names depend on the dataset version; we try the
    # documented forms and log any 404s.
    candidates = [
        ("fineweb2",  {"path": "HuggingFaceFW/fineweb-2",
                       "name": f"{lang_code}_Latn", "split": "train",
                       "text_key": "text"}),
        ("wura",      {"path": "castorini/wura",
                       "name": lang_code, "split": "train",
                       "text_key": "content"}),
        ("madlad400", {"path": "allenai/MADLAD-400",
                       "name": lang_code, "split": "clean",
                       "text_key": "text"}),
    ]

    for source_name, kwargs in candidates:
        text_key = kwargs.pop("text_key")
        try:
            def _make_iter(kw=kwargs, tk=text_key):
                ds = load_dataset(streaming=True, **kw)
                for ex in ds:
                    yield ex.get(tk) or ""
            # Probe the first document to confirm the config exists.
            it = _make_iter()
            first = next(it)
            def factory(kw=kwargs, tk=text_key, _first=first):
                # Yield the probe doc first, then a fresh stream skipping it
                # would be wasteful — easier: just start a fresh stream.
                ds = load_dataset(streaming=True, **kw)
                for ex in ds:
                    yield ex.get(tk) or ""
            log.info(f"  ✓ {lang_code}: using {source_name} ({kwargs.get('name')})")
            return factory
        except Exception as e:
            log.debug(f"  ✗ {lang_code}: {source_name} not available ({type(e).__name__}: {e})")
            continue

    log.warning(f"  ⚠ {lang_code}: no monolingual source available — skipping")
    return None


def load_code_stream(log: logging.Logger):
    """CornStack-Python — code component."""
    from datasets import load_dataset
    def factory():
        ds = load_dataset("nomic-ai/cornstack-python-v1",
                          streaming=True, split="train")
        for ex in ds:
            # CornStack rows expose `content` (the source file body).
            txt = ex.get("content") or ex.get("code") or ex.get("text") or ""
            if txt:
                yield txt
    try:
        next(factory())
        log.info("  ✓ code: CornStack-Python")
        return factory
    except Exception as e:
        log.warning(f"  ⚠ code: CornStack unavailable ({e}) — skipping")
        return None


def load_math_stream(log: logging.Logger):
    """FineMath-4+ — math component."""
    from datasets import load_dataset
    def factory():
        ds = load_dataset("HuggingFaceTB/finemath", "finemath-4plus",
                          streaming=True, split="train")
        for ex in ds:
            yield ex.get("text") or ""
    try:
        next(factory())
        log.info("  ✓ math: FineMath-4+")
        return factory
    except Exception as e:
        log.warning(f"  ⚠ math: FineMath-4+ unavailable ({e}) — skipping")
        return None


def load_synthetic_stream(lang_code: str, synthetic_repo: Optional[str], log: logging.Logger):
    """
    Synthetic component. The spec describes GPT-4.1 translations across 11
    domains, but the actual artefact is not a public HF dataset. The user
    supplies `--synthetic-repo` pointing at their own preprocessed dataset,
    or this component is skipped (which is fine for the smoke test).
    """
    if not synthetic_repo:
        return None
    from datasets import load_dataset
    def factory():
        try:
            ds = load_dataset(synthetic_repo, lang_code,
                              streaming=True, split="train")
        except Exception:
            # Try without a language config — assume a 'lang' column.
            ds = load_dataset(synthetic_repo, streaming=True, split="train")
            ds = ds.filter(lambda ex: ex.get("lang") == lang_code)
        for ex in ds:
            yield ex.get("text") or ex.get("content") or ""
    try:
        next(factory())
        return factory
    except Exception as e:
        log.debug(f"  ✗ synthetic/{lang_code}: {e}")
        return None


# ---------------------------------------------------------------------------
# 4. Main curation pipeline.
# ---------------------------------------------------------------------------

@dataclass
class CurationStats:
    """Bookkeeping for the manifest written to disk at the end."""
    component_actual_tokens: dict[str, int] = field(default_factory=dict)
    per_lang_actual_tokens: dict[str, dict[str, int]] = field(default_factory=dict)
    documents_written: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0


def curate(args: argparse.Namespace) -> None:
    """Top-level curation pipeline."""
    log = _setup_logging(args.output_dir)
    log.info("=" * 70)
    log.info("AfriqueGemma CMS curation")
    log.info("=" * 70)

    budget = build_budget(args.total_tokens)
    _log_budget(budget, log)

    # Import heavy deps lazily so --help is fast and the script's failure
    # mode is informative when a dependency is missing.
    try:
        from transformers import AutoTokenizer
        from datasets import Dataset  # noqa: F401  (verified at import time)
    except ImportError as e:
        log.error(f"Missing dependency: {e}. Install with: pip install -r requirements.txt")
        sys.exit(1)

    # Resolve the tokenizer choice from the registry.
    tokenizer_repo = TOKENIZER_REGISTRY[args.tokenizer]
    log.info(f"Loading tokenizer: {args.tokenizer} → {tokenizer_repo}")
    if args.tokenizer != DEFAULT_TOKENIZER_KEY:
        log.warning(
            f"You selected --tokenizer {args.tokenizer}. The token budgets in "
            f"the AfriqueGemma spec are measured with the Gemma 3 tokenizer, "
            f"so the volume of underlying text will differ from what the "
            f"paper used. This is the right behaviour for a fair comparison "
            f"across models (each model sees the same number of its own "
            f"tokens), but be aware of it when reading the manifest."
        )

    if "gemma" in tokenizer_repo.lower() and not os.environ.get("HF_TOKEN"):
        log.warning("HF_TOKEN is not set. Gemma 3 tokenizer is gated — "
                    "this script will fail when it tries to download it. "
                    "Get a token at https://huggingface.co/settings/tokens "
                    "and accept the Gemma 3 license at "
                    f"https://huggingface.co/{tokenizer_repo}.")
    if "llama" in tokenizer_repo.lower() and not os.environ.get("HF_TOKEN"):
        log.warning("HF_TOKEN is not set. LLaMA tokenizers are gated — "
                    "this script will fail when it tries to download. "
                    f"Accept the licence at https://huggingface.co/{tokenizer_repo}.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)

    stats = CurationStats(started_at=time.time())
    out_train = Path(args.output_dir) / "train_raw.jsonl"
    out_train.parent.mkdir(parents=True, exist_ok=True)

    # We write JSONL streaming, one {"text": ...} per line. Peak RAM stays
    # bounded regardless of corpus size because nothing is buffered in memory.
    rng = random.Random(args.seed)

    with out_train.open("w", encoding="utf-8") as fout:
        _curate_monolingual(budget, tokenizer, fout, stats, rng, args, log)
        _curate_code(budget, tokenizer, fout, stats, args, log)
        _curate_math(budget, tokenizer, fout, stats, args, log)
        _curate_synthetic(budget, tokenizer, fout, stats, args, log)

    stats.finished_at = time.time()
    log.info(f"Wrote {stats.documents_written:,} documents to {out_train}")

    # Stage the final layout for the trainer.
    _finalize(out_train, Path(args.output_dir), log)

    # Manifest for reproducibility.
    _write_manifest(args, budget, stats, log)
    log.info("Done.")


def _log_budget(budget: CurationBudget, log: logging.Logger) -> None:
    log.info(f"Total token budget: {budget.total_tokens:,}")
    log.info("Component budgets:")
    for comp, n in budget.component_tokens.items():
        pct = 100 * n / budget.total_tokens
        log.info(f"  {comp:14s} {n:>15,}  ({pct:5.2f}%)")
    log.info(f"Monolingual per-language budgets:")
    for lang, n in budget.monolingual_per_lang.items():
        tag = "HRL" if lang in HIGH_RESOURCE_LANGUAGES else "AFR"
        log.info(f"  [{tag}] {lang}: {n:,}")
    log.info(f"Synthetic per-language budgets ({len(budget.synthetic_per_lang)} langs):")
    for lang, n in budget.synthetic_per_lang.items():
        log.info(f"  [SYN] {lang}: {n:,}")


def _curate_monolingual(budget, tokenizer, fout, stats, rng, args, log):
    log.info("-" * 70)
    log.info("Component: monolingual")
    log.info("-" * 70)
    stats.component_actual_tokens["monolingual"] = 0
    stats.per_lang_actual_tokens["monolingual"] = {}

    for lang, quota in budget.monolingual_per_lang.items():
        factory = load_monolingual_stream("auto", lang, log)
        if factory is None:
            stats.per_lang_actual_tokens["monolingual"][lang] = 0
            continue

        used = 0
        # UniMax: up to 5 passes for languages that exhaust their source.
        max_passes = UNIMAX_MAX_EPOCHS if lang in AFRICAN_LANGUAGES else 1
        for pass_idx in range(max_passes):
            it = factory()
            remaining = quota - used
            if remaining <= 0:
                break
            consumed_this_pass = 0
            for doc, n_tok in stream_until_quota(
                it, remaining, tokenizer,
                source_label=f"mono/{lang} pass {pass_idx+1}",
                log=log,
            ):
                fout.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
                used += n_tok
                consumed_this_pass += n_tok
                stats.documents_written += 1
                if used >= quota:
                    break
            if consumed_this_pass == 0:
                # Source exhausted without yielding anything new — bail.
                break
            if used >= quota:
                break

        stats.per_lang_actual_tokens["monolingual"][lang] = used
        stats.component_actual_tokens["monolingual"] += used


def _curate_code(budget, tokenizer, fout, stats, args, log):
    log.info("-" * 70)
    log.info("Component: code")
    log.info("-" * 70)
    quota = budget.component_tokens["code"]
    factory = load_code_stream(log)
    if factory is None:
        stats.component_actual_tokens["code"] = 0
        return
    used = 0
    for doc, n_tok in stream_until_quota(factory(), quota, tokenizer,
                                         source_label="code", log=log):
        fout.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
        used += n_tok
        stats.documents_written += 1
        if used >= quota:
            break
    stats.component_actual_tokens["code"] = used


def _curate_math(budget, tokenizer, fout, stats, args, log):
    log.info("-" * 70)
    log.info("Component: math")
    log.info("-" * 70)
    quota = budget.component_tokens["math"]
    factory = load_math_stream(log)
    if factory is None:
        stats.component_actual_tokens["math"] = 0
        return
    used = 0
    for doc, n_tok in stream_until_quota(factory(), quota, tokenizer,
                                         source_label="math", log=log):
        fout.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
        used += n_tok
        stats.documents_written += 1
        if used >= quota:
            break
    stats.component_actual_tokens["math"] = used


def _curate_synthetic(budget, tokenizer, fout, stats, args, log):
    log.info("-" * 70)
    log.info("Component: synthetic")
    log.info("-" * 70)
    if not args.synthetic_repo:
        log.info("  No --synthetic-repo supplied; skipping synthetic component. "
                 "This is expected for smoke tests.")
        stats.component_actual_tokens["synthetic"] = 0
        return
    stats.component_actual_tokens["synthetic"] = 0
    stats.per_lang_actual_tokens["synthetic"] = {}
    for lang, quota in budget.synthetic_per_lang.items():
        factory = load_synthetic_stream(lang, args.synthetic_repo, log)
        if factory is None:
            stats.per_lang_actual_tokens["synthetic"][lang] = 0
            continue
        used = 0
        for doc, n_tok in stream_until_quota(
            factory(), quota, tokenizer,
            source_label=f"syn/{lang}", log=log,
        ):
            fout.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
            used += n_tok
            stats.documents_written += 1
            if used >= quota:
                break
        stats.per_lang_actual_tokens["synthetic"][lang] = used
        stats.component_actual_tokens["synthetic"] += used


def _finalize(jsonl_path: Path, out_dir: Path,
              log: logging.Logger) -> None:
    """
    Stage the streamed JSONL into the final layout expected by LLaMA-Factory.

    The corpus is written as a single JSONL file (`train.jsonl`) with one
    `{"text": ...}` document per line. JSONL is LLaMA-Factory's most reliable
    pretraining input path; it also keeps peak memory bounded regardless of
    corpus size, which matters at the full 25.2B-token scale.

    No validation slice is produced. Matches the reference paper, which
    tracks training loss only and validates on external benchmarks. If you
    want an eval-loss curve during training, add `val_size: 0.001` to your
    LLaMA-Factory YAML and it will split the training stream at load time.
    """
    # Rename the staging file to its canonical name.
    final_path = out_dir / "train.jsonl"
    if jsonl_path.resolve() != final_path.resolve():
        jsonl_path.rename(final_path)
    size_mb = final_path.stat().st_size / 1_000_000
    log.info(f"Final dataset: {final_path} ({size_mb:,.1f} MB)")

    # Write the LLaMA-Factory dataset registration stub. Merge the contents
    # of this file into LLaMA-Factory's `data/dataset_info.json`, then set
    #
    #   dataset_dir: <out_dir>
    #   dataset: afriquegemma_cms
    #
    # in your training YAML. Add `val_size: 0.001` there if you want
    # eval-loss tracking — see notes in CurationBudget.
    registration = {
        "afriquegemma_cms": {
            # `load_from: file` + `file_name` is LLaMA-Factory's
            # canonical way to load a local JSONL pretraining corpus.
            "load_from": "file",
            "file_name": "train.jsonl",
            "columns": {"prompt": "text"},
        }
    }
    with (out_dir / "dataset_info.json").open("w") as f:
        json.dump(registration, f, indent=2)
    log.info(f"Wrote LLaMA-Factory registration: {out_dir / 'dataset_info.json'}")


def _write_manifest(args, budget, stats, log):
    out = Path(args.output_dir) / "curation_manifest.json"
    manifest = {
        "preset": args.preset,
        "total_tokens_requested": args.total_tokens,
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "tokenizer_repo": TOKENIZER_REGISTRY[args.tokenizer],
        "budget": asdict(budget),
        "actual": {
            "component_tokens": stats.component_actual_tokens,
            "per_language_tokens": stats.per_lang_actual_tokens,
            "documents_written": stats.documents_written,
        },
        "wallclock_seconds": stats.finished_at - stats.started_at,
        "sources_attempted": SOURCE_DATASETS,
    }
    with out.open("w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Wrote manifest: {out}")

    # One-line summary.
    actual_total = sum(stats.component_actual_tokens.values())
    target = budget.total_tokens
    log.info(f"Actual total tokens: {actual_total:,} "
             f"({100 * actual_total / target:.1f}% of target {target:,})")


def _setup_logging(out_dir: str) -> logging.Logger:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("curate_cms")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                            datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    fh = logging.FileHandler(Path(out_dir) / "curation.log")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


# ---------------------------------------------------------------------------
# 5. CLI.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Curate the AfriqueGemma CMS corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--preset", choices=["full", "smoke"], default="smoke",
                   help="'full' = ~25.2B tokens (the paper); "
                        "'smoke' = ~50M tokens, same proportions, for "
                        "end-to-end pipeline testing.")
    p.add_argument("--total-tokens", type=int, default=None,
                   help="Override the total token budget. Component and "
                        "per-language ratios are preserved. If omitted, "
                        "the preset's default is used.")
    p.add_argument("--output-dir", required=True,
                   help="Destination directory on scratch.")
    p.add_argument("--synthetic-repo", default=None,
                   help="Optional HF repo (or local path) for the synthetic "
                        "component. If omitted, S is skipped — fine for "
                        "smoke tests.")
    p.add_argument("--tokenizer", choices=sorted(TOKENIZER_REGISTRY.keys()),
                   default=DEFAULT_TOKENIZER_KEY,
                   help="Which tokenizer to use for token-counting. The "
                        "spec's budgets are defined with the Gemma 3 "
                        "tokenizer; using a different one preserves text "
                        "proportions but changes total text volume. To add "
                        "a new tokenizer, edit TOKENIZER_REGISTRY at the "
                        "top of this file.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.total_tokens is None:
        args.total_tokens = (FULL_TOTAL_TOKENS if args.preset == "full"
                             else SMOKE_TOTAL_TOKENS)
    return args


if __name__ == "__main__":
    curate(parse_args())
