#!/usr/bin/env python3
"""
Agrega **todas** as respostas type2 (extraction_*.json) por modelo da dissertação
e calcula, para cada um: n, média, mínimo, Q1, mediana, Q3, máximo.

Métricas:
  - **utf8_bytes**: len do JSON canónico `{"entities":[...]}` em bytes UTF-8
    (independente do tokenizer).
  - **tokens**: `len(tokenizer.encode(..., add_special_tokens=False))` com o
    tokenizer **desse** modelo (como a geração é contada por cada sistema).

Saída: reports/token_analysis/response_size_quartiles_by_model.json

Uso (raiz onconavegation-BioNER_llm):

  .venv_plot/bin/python scripts/token_analysis/response_size_quartiles_by_model.py
  .venv_plot/bin/python scripts/token_analysis/response_size_quartiles_by_model.py --no-tokens
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from huggingface_hub import login as hf_hub_login
from transformers import AutoTokenizer

from token_analysis.registry import (
    DISSERTATION_LABELS,
    DISSERTATION_MODEL_HF_IDS,
    tokenizer_repo_id,
)

warnings.filterwarnings("ignore", message=".*sequence length is longer than.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

SLUG_TO_HF: Dict[str, str] = {
    "meta-llama-3.1-70b-instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "qwen2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
    "yi-1.5-9b-chat": "01-ai/Yi-1.5-9B-Chat",
    "meta-llama-3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3-openbiollm-8b": "aaditya/Llama3-OpenBioLLM-8B",
    "internlm2-5-1-8b-chat": "internlm/internlm2_5-1_8b-chat",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "mistral-7b-instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "internlm2-5-7b-chat": "internlm/internlm2_5-7b-chat",
    "h2o-danube3-4b-chat": "h2oai/h2o-danube3-4b-chat",
    "phi-3-mini-128k-instruct": "microsoft/Phi-3-mini-128k-instruct",
    "phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",
    "llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "gemma-1.1-2b-it": "google/gemma-1.1-2b-it",
    "smollm2-1.7b-instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "qwen2-1.5b-instruct": "Qwen/Qwen2-1.5B-Instruct",
    "llama-3.2-1b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
}

HF_TO_SLUG: Dict[str, str] = {v: k for k, v in SLUG_TO_HF.items()}


def entities_from_extraction(data: Dict[str, Any]) -> List[Dict[str, str]]:
    raw = data.get("entities")
    out: List[Dict[str, str]] = []
    if isinstance(raw, list):
        for e in raw:
            if isinstance(e, dict) and e.get("text"):
                out.append(
                    {
                        "text": str(e.get("text", "")),
                        "type": str(e.get("type", "")),
                    }
                )
        return out
    if isinstance(raw, dict):
        for c in raw.get("chemicals") or []:
            if isinstance(c, dict) and c.get("text"):
                out.append({"text": str(c["text"]), "type": "Chemical"})
        for d in raw.get("diseases") or []:
            if isinstance(d, dict) and d.get("text"):
                out.append({"text": str(d["text"]), "type": "Disease"})
    return out


def canonical_response_json(entities: List[Dict[str, str]]) -> str:
    return json.dumps({"entities": entities}, ensure_ascii=False, separators=(",", ":"))


def resolve_slug(path: Path) -> Optional[str]:
    parts = path.parts
    try:
        idx = parts.index("indicios_encontrados")
        slug = parts[idx + 1]
    except (ValueError, IndexError):
        slug = None
        for p in parts:
            if p in SLUG_TO_HF:
                slug = p
                break
    if slug is None or slug not in SLUG_TO_HF:
        return None
    return slug


def summarize(values: List[int]) -> Dict[str, float]:
    """n, mean, min, q1, median, q3, max (linear interpolation, same spirit as numpy)."""
    if not values:
        return {}
    try:
        import numpy as np

        arr = np.asarray(values, dtype=np.float64)
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(np.min(arr)),
            "q1": float(np.percentile(arr, 25)),
            "median": float(np.percentile(arr, 50)),
            "q3": float(np.percentile(arr, 75)),
            "max": float(np.max(arr)),
        }
    except ImportError:
        xs = sorted(values)
        n = len(xs)
        mean = sum(xs) / n

        def pct(p: float) -> float:
            if n == 1:
                return float(xs[0])
            k = (n - 1) * p / 100.0
            lo = int(k)
            hi = min(lo + 1, n - 1)
            w = k - lo
            return xs[lo] * (1 - w) + xs[hi] * w

        return {
            "n": n,
            "mean": mean,
            "std": float(
                (sum((x - mean) ** 2 for x in xs) / n) ** 0.5
            ),
            "min": float(xs[0]),
            "q1": pct(25),
            "median": pct(50),
            "q3": pct(75),
            "max": float(xs[-1]),
        }


def load_tokenizers(
    hf_token: Optional[str], local_only: bool
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """slug -> tokenizer; errors slug -> message."""
    tokenizers: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    for hf_id in DISSERTATION_MODEL_HF_IDS:
        slug = HF_TO_SLUG.get(hf_id)
        if not slug:
            continue
        tok_id = tokenizer_repo_id(hf_id)
        try:
            tok_kw: Dict[str, Any] = {
                "trust_remote_code": True,
                "local_files_only": local_only,
                "token": hf_token,
            }
            if hf_id.startswith("internlm/"):
                tok_kw["use_fast"] = False
            tokenizers[slug] = AutoTokenizer.from_pretrained(tok_id, **tok_kw)
        except Exception as e:
            errors[slug] = f"{hf_id} ({tok_id}): {e}"
    return tokenizers, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indicios-dir",
        type=Path,
        default=ROOT / "indicios_encontrados",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "reports"
        / "token_analysis"
        / "response_size_quartiles_by_model.json",
    )
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument(
        "--no-tokens",
        action="store_true",
        help="Só bytes UTF-8 (não carrega tokenizers nem faz encode).",
    )
    args = parser.parse_args()

    hf_token = (
        args.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    if hf_token:
        hf_hub_login(token=hf_token, add_to_git_credential=False)

    if not args.indicios_dir.is_dir():
        print(f"Pasta não encontrada: {args.indicios_dir}")
        return 1

    tokenizers: Dict[str, Any] = {}
    tok_errors: Dict[str, str] = {}
    if not args.no_tokens:
        print("A carregar tokenizers...")
        tokenizers, tok_errors = load_tokenizers(hf_token, args.local_only)
        if tok_errors:
            for slug, msg in sorted(tok_errors.items()):
                print(f"  AVISO tokenizer {slug}: {msg}")

    by_slug_bytes: Dict[str, List[int]] = {s: [] for s in SLUG_TO_HF}
    by_slug_tokens: Dict[str, List[int]] = {s: [] for s in SLUG_TO_HF}

    print("A varrer extraction_*.json em */type2/ ...")
    n_ok = 0
    n_skip = 0
    for path in sorted(args.indicios_dir.rglob("extraction_*.json")):
        if "type2" not in path.parts:
            continue
        slug = resolve_slug(path)
        if slug is None:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            n_skip += 1
            continue

        entities = entities_from_extraction(data)
        text = canonical_response_json(entities)
        b = len(text.encode("utf-8"))
        by_slug_bytes[slug].append(b)
        tok = tokenizers.get(slug)
        if tok is not None:
            by_slug_tokens[slug].append(
                len(tok.encode(text, add_special_tokens=False))
            )
        n_ok += 1
        if n_ok % 25000 == 0:
            print(f"  ... {n_ok} ficheiros")

    print(f"  Total válidos: {n_ok}  ignorados: {n_skip}")

    per_model: List[Dict[str, Any]] = []
    for hf_id in DISSERTATION_MODEL_HF_IDS:
        slug = HF_TO_SLUG.get(hf_id)
        label = DISSERTATION_LABELS.get(hf_id, hf_id)
        if not slug:
            continue
        row: Dict[str, Any] = {
            "hf_id": hf_id,
            "label_short": label,
            "folder_slug": slug,
            "utf8_bytes": summarize(by_slug_bytes.get(slug, [])),
        }
        if not args.no_tokens and slug in tokenizers:
            row["tokens"] = summarize(by_slug_tokens.get(slug, []))
        elif not args.no_tokens and slug in tok_errors:
            row["tokens"] = {"error": tok_errors[slug]}
        per_model.append(row)

    out_doc = {
        "description": (
            "Quartis da dimensão da resposta (JSON canónico type2) por modelo. "
            "utf8_bytes: tamanho do ficheiro lógico em bytes; tokens: encode desse "
            "texto com o tokenizer do modelo que gerou os indícios."
        ),
        "no_tokens_flag": args.no_tokens,
        "files_total_valid": n_ok,
        "files_skipped": n_skip,
        "by_model": per_model,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, indent=2, ensure_ascii=False)

    print(f"\nRelatório: {args.out}\n")
    hdr = f"{'modelo':<22} {'n':>7} {'mean_B':>8} {'Q1_B':>7} {'med_B':>7} {'Q3_B':>7} {'mean_T':>8}"
    print(hdr)
    print("-" * len(hdr))
    for row in per_model:
        u = row.get("utf8_bytes") or {}
        t = row.get("tokens") or {}
        mean_t = ""
        if isinstance(t, dict) and "mean" in t:
            mean_t = f"{t['mean']:.1f}"
        elif isinstance(t, dict) and "error" in t:
            mean_t = "ERR"
        print(
            f"{row['label_short']:<22} {int(u.get('n', 0)):>7} "
            f"{u.get('mean', 0):>8.1f} {u.get('q1', 0):>7.1f} {u.get('median', 0):>7.1f} "
            f"{u.get('q3', 0):>7.1f} {mean_t:>8}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
