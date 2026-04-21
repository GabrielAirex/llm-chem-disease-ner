#!/usr/bin/env python3
"""
Reconstrói o prompt type2 de cada extraction_*.json (PromptEngine + config.yaml,
como a API) e conta tokens de **entrada** com `count_prompt` (chat template +
add_generation_prompt), usando o tokenizer de **cada** modelo da dissertação.

Agrega quartis por pasta modelo (indicios_encontrados).

Saída: reports/token_analysis/prompt_input_quartiles_by_model.json

Uso (raiz onconavegation-BioNER_llm):

  .venv_plot/bin/python scripts/token_analysis/prompt_input_quartiles_by_model.py
  .venv_plot/bin/python scripts/token_analysis/plot.py --prompt-quartiles
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

import yaml
from huggingface_hub import login as hf_hub_login

from token_analysis.count import count_prompt
from token_analysis.registry import DISSERTATION_LABELS, DISSERTATION_MODEL_HF_IDS
from token_analysis.response_size_quartiles_by_model import (
    HF_TO_SLUG,
    SLUG_TO_HF,
    load_tokenizers,
    resolve_slug,
    summarize,
)

from src.models.schemas import PromptStrategy
from src.prompts.prompt_engine import PromptEngine

warnings.filterwarnings("ignore", message=".*sequence length is longer than.*")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("src.prompts.prompt_engine").setLevel(logging.ERROR)


def strategy_and_k(data: Dict[str, Any]) -> tuple[PromptStrategy, Optional[int]]:
    ps = (data.get("prompt_strategy") or "few-shot").strip().lower()
    if ps == "zero-shot":
        return PromptStrategy.ZERO_SHOT, None
    ne = data.get("num_examples")
    if ne is None:
        return PromptStrategy.FEW_SHOT, 3
    try:
        return PromptStrategy.FEW_SHOT, int(ne)
    except (TypeError, ValueError):
        return PromptStrategy.FEW_SHOT, 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indicios-dir",
        type=Path,
        default=ROOT / "indicios_encontrados",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "config.yaml",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "reports"
        / "token_analysis"
        / "prompt_input_quartiles_by_model.json",
    )
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=None,
        help="Override de prompts.max_text_length (default: lê do config)",
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

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    prompts_cfg = cfg.get("prompts") or {}
    max_text_length = (
        int(args.max_text_length)
        if args.max_text_length is not None
        else int(prompts_cfg.get("max_text_length", 4000))
    )
    use_positions = bool(prompts_cfg.get("use_positions", False))
    sep_models_cfg = (cfg.get("chat_template_separation") or {}).get("models") or {}

    engine = PromptEngine(prompt_type="type2")

    print("A carregar tokenizers...")
    tokenizers, tok_errors = load_tokenizers(hf_token, args.local_only)
    if tok_errors:
        for slug, msg in sorted(tok_errors.items()):
            print(f"  AVISO tokenizer {slug}: {msg}")

    by_slug: Dict[str, List[int]] = {s: [] for s in SLUG_TO_HF}
    n_ok = 0
    n_skip = 0
    n_prompt_err = 0

    print("A varrer extraction_*.json em */type2/ ...")
    for path in sorted(args.indicios_dir.rglob("extraction_*.json")):
        if "type2" not in path.parts:
            continue
        slug = resolve_slug(path)
        if slug is None:
            continue
        tok = tokenizers.get(slug)
        if tok is None:
            n_skip += 1
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            n_skip += 1
            continue

        text = data.get("text") or ""
        strat, num_ex = strategy_and_k(data)
        try:
            prompt = engine.generate_prompt(
                text=text,
                strategy=strat,
                use_positions=use_positions,
                num_examples=num_ex,
                max_text_length=max_text_length,
            )
        except Exception:
            n_prompt_err += 1
            n_skip += 1
            continue

        hf_id = SLUG_TO_HF[slug]
        use_sep = bool(sep_models_cfg.get(hf_id))
        try:
            n_tok = int(count_prompt(tok, prompt, use_sep))
        except Exception:
            n_skip += 1
            continue

        by_slug[slug].append(n_tok)
        n_ok += 1
        if n_ok % 25000 == 0:
            print(f"  ... {n_ok} ficheiros")

    print(
        f"  Total válidos: {n_ok}  ignorados/erros: {n_skip}  "
        f"(erros generate_prompt: {n_prompt_err})"
    )

    per_model: List[Dict[str, Any]] = []
    for hf_id in DISSERTATION_MODEL_HF_IDS:
        slug = HF_TO_SLUG.get(hf_id)
        label = DISSERTATION_LABELS.get(hf_id, hf_id)
        if not slug:
            continue
        stats = summarize(by_slug.get(slug, []))
        row: Dict[str, Any] = {
            "hf_id": hf_id,
            "label_short": label,
            "folder_slug": slug,
            "tokens": stats,
        }
        if slug in tok_errors:
            row["tokenizer_error"] = tok_errors[slug]
        per_model.append(row)

    out_doc: Dict[str, Any] = {
        "description": (
            "Quartis dos tokens de **entrada** do prompt type2 (reconstruído com "
            "PromptEngine + config.yaml), por modelo. Contagem = count_prompt "
            "(apply_chat_template com add_generation_prompt), alinhado a analyze.py."
        ),
        "config_ref": {
            "config_path": str(args.config),
            "max_text_length": max_text_length,
            "use_positions": use_positions,
            "chat_template_separation_models": sep_models_cfg,
        },
        "files_total_valid": n_ok,
        "files_skipped_or_failed": n_skip,
        "by_model": per_model,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, indent=2, ensure_ascii=False)

    print(f"\nRelatório: {args.out}\n")
    hdr = f"{'modelo':<22} {'n':>7} {'mean':>8} {'Q1':>7} {'med':>7} {'Q3':>7} {'max':>7}"
    print(hdr)
    print("-" * len(hdr))
    for row in per_model:
        t = row.get("tokens") or {}
        if not t:
            print(f"{row['label_short']:<22} (sem dados)")
            continue
        print(
            f"{row['label_short']:<22} {int(t.get('n', 0)):>7} "
            f"{t.get('mean', 0):>8.1f} {t.get('q1', 0):>7.1f} {t.get('median', 0):>7.1f} "
            f"{t.get('q3', 0):>7.1f} {t.get('max', 0):>7.1f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
