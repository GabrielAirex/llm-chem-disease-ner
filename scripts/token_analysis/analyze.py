#!/usr/bin/env python3
"""
Análise de tokens do zero: cada modelo nos scripts vLLM usa o seu tokenizer HF.

Cenário:
  - Artigo BC5CDR com maior campo `text` (cdr_gold.csv).
  - Prompts type2 para k ∈ {0,1,2,4,8,16,32} (PromptEngine).
  - Por modelo: contagens com apply_chat_template; total = entrada + max_tokens (config).

Gera JSON estável (schema_version) em reports/token_analysis/ para o plot.py consumir.

Uso (raiz do projeto onconavegation-BioNER_llm):

  python3 scripts/token_analysis/analyze.py
  HF_TOKEN=hf_... python3 scripts/token_analysis/analyze.py
  # ou: HUGGING_FACE_HUB_TOKEN / HUGGINGFACE_HUB_TOKEN
  python3 scripts/token_analysis/analyze.py --only-models Qwen/Qwen2.5-3B-Instruct
  python3 scripts/token_analysis/analyze.py --all-vllm-models   # todos os MODEL= nos .sh
  python3 scripts/token_analysis/analyze.py --output reports/token_analysis/meus_dados.json

Por defeito: só os 18 modelos da tabela da dissertação (registry.DISSERTATION_MODEL_HF_IDS).

Dependências úteis: pip install protobuf sentencepiece tiktoken (InternLM/Qwen precisam de protobuf para o SentencePiece).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
# Pacote `token_analysis` vive em scripts/token_analysis/; o dir do .py não pode ser o único no path.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

import yaml
from huggingface_hub import login as hf_hub_login
from transformers import AutoTokenizer

from token_analysis import SCHEMA_VERSION
from token_analysis.count import count_prompt
from token_analysis.registry import (
    DISSERTATION_LABELS,
    DISSERTATION_MODEL_HF_IDS,
    apply_context_max_override,
    effective_context_window,
    parse_vllm_registry,
    synthetic_registry_entry,
    tokenizer_repo_id,
)

from src.models.schemas import PromptStrategy
from src.prompts.prompt_engine import PromptEngine

warnings.filterwarnings("ignore", message=".*sequence length is longer than.*")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("src.prompts.prompt_engine").setLevel(logging.ERROR)

KS_DEFAULT = [0, 1, 2, 4, 8, 16, 32]
K_WORST = 32


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_longest_text_row(rows: List[Dict[str, str]]) -> Dict[str, str]:
    return max(rows, key=lambda r: len((r.get("text") or "").strip()))


def short_label(hf_id: str) -> str:
    return DISSERTATION_LABELS.get(hf_id) or (
        hf_id.split("/", 1)[-1] if "/" in hf_id else hf_id
    )


def build_prompts_by_k(
    engine: PromptEngine,
    text: str,
    max_text_length: int,
    ks: List[int],
) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for k in ks:
        strat = PromptStrategy.ZERO_SHOT if k == 0 else PromptStrategy.FEW_SHOT
        out[k] = engine.generate_prompt(
            text=text,
            strategy=strat,
            use_positions=False,
            num_examples=k,
            max_text_length=max_text_length,
        )
    return out


def run() -> int:
    parser = argparse.ArgumentParser(description="Análise de tokens → JSON para plot")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "config.yaml")
    parser.add_argument("--csv", type=Path, default=ROOT / "src" / "preprocessing" / "cdr_gold.csv")
    parser.add_argument("--vllm-dir", type=Path, default=ROOT / "vllm")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON de saída (default: reports/token_analysis/report_<ts>.json)",
    )
    parser.add_argument("--only-models", nargs="*", default=None)
    parser.add_argument(
        "--all-vllm-models",
        action="store_true",
        help="Analisar todos os MODEL= nos scripts vLLM (ignora lista da dissertação)",
    )
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--hf-token", default=None)
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

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    max_text_length = (
        int(args.max_text_length)
        if args.max_text_length is not None
        else int(cfg.get("prompts", {}).get("max_text_length", 4000))
    )
    max_new_tokens = int(cfg.get("llm_defaults", {}).get("max_tokens", 600))
    sep_cfg = cfg.get("chat_template_separation", {}) or {}
    sep_models_cfg = sep_cfg.get("models") or {}

    registry, reg_warnings = parse_vllm_registry(args.vllm_dir)

    if args.only_models:
        hf_ids = list(args.only_models)
        missing_reg = [h for h in hf_ids if h not in registry]
        if missing_reg:
            print("Aviso: estes IDs não têm .sh vLLM; uso janela 16k sintética:", missing_reg)
    elif args.all_vllm_models:
        hf_ids = sorted(registry.keys())
    else:
        hf_ids = list(DISSERTATION_MODEL_HF_IDS)
        missing_reg = [h for h in hf_ids if h not in registry]
        if missing_reg:
            print("Aviso: sem entrada vLLM (janela 16k sintética):", missing_reg)

    rows = load_rows(args.csv)
    longest = pick_longest_text_row(rows)
    raw_text = longest.get("text") or ""
    pmid = str(longest.get("pmid", ""))

    engine = PromptEngine(prompt_type="type2")
    prompts_by_k = build_prompts_by_k(engine, raw_text, max_text_length, KS_DEFAULT)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "reports" / "token_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output or (out_dir / f"report_{ts}.json")

    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": ts,
        "meta": {
            "project_root": str(ROOT),
            "csv": str(args.csv),
            "longest_pmid": pmid,
            "longest_text_chars": len(raw_text),
            "max_text_length_chars": max_text_length,
            "max_new_tokens": max_new_tokens,
            "prompt_type": "type2",
            "ks": KS_DEFAULT,
            "k_reference_worst_case": K_WORST,
            "effective_window_rule": (
                "min(16384, max_model_len_max); overrides em registry.VLLM_CONTEXT_MAX_OVERRIDES"
            ),
            "model_set": (
                "all_vllm"
                if args.all_vllm_models
                else ("custom" if args.only_models else "dissertation_18")
            ),
            "dissertation_model_order": (
                list(DISSERTATION_MODEL_HF_IDS)
                if not args.all_vllm_models and not args.only_models
                else None
            ),
        },
        "registry_warnings": reg_warnings,
        "models": {},
    }

    print(
        f"Maior texto: pmid={pmid} |text|={len(raw_text)} | "
        f"max_text_length={max_text_length} | max_new_tokens={max_new_tokens}"
    )

    for hf_id in hf_ids:
        base = registry.get(hf_id) or synthetic_registry_entry(hf_id)
        info = apply_context_max_override(hf_id, base)
        tok_id = tokenizer_repo_id(hf_id)
        use_sep = bool(sep_models_cfg.get(hf_id))
        window = effective_context_window(info)
        vllm_max = int(info["max_model_len_max"])
        vllm_min = int(info["max_model_len_min"])

        entry: Dict[str, Any] = {
            "hf_id": hf_id,
            "label_short": short_label(hf_id),
            "tokenizer_hf_id": tok_id,
            "vllm_max_model_len": vllm_max,
            "vllm_min_model_len": vllm_min,
            "vllm_scripts": info["scripts"],
            "chat_template_separation": use_sep,
            "effective_window": window,
        }
        if "max_model_len_max_from_script" in info:
            entry["vllm_max_model_len_from_script_before_override"] = info[
                "max_model_len_max_from_script"
            ]

        try:
            tok_kw: Dict[str, Any] = {
                "trust_remote_code": True,
                "local_files_only": args.local_only,
                "token": hf_token,
            }
            # SentencePiece “fast” por vezes falha com tokenizer.model (ex.: InternLM).
            if hf_id.startswith("internlm/"):
                tok_kw["use_fast"] = False

            tokenizer = AutoTokenizer.from_pretrained(tok_id, **tok_kw)
        except Exception as e:
            entry["status"] = "tokenizer_error"
            entry["error"] = str(e)
            report["models"][hf_id] = entry
            print(f"[erro tokenizer] {hf_id}: {e}")
            continue

        by_k_input: Dict[str, int] = {}
        by_k_total: Dict[str, int] = {}
        for k in KS_DEFAULT:
            n_in = count_prompt(tokenizer, prompts_by_k[k], use_sep)
            n_tot = n_in + max_new_tokens
            by_k_input[str(k)] = n_in
            by_k_total[str(k)] = n_tot

        n32_in = by_k_input[str(K_WORST)]
        n32_tot = by_k_total[str(K_WORST)]
        fits_win = n32_tot <= window
        fits_16k = n32_tot <= 16384

        entry["status"] = "ok"
        entry["tokens"] = {
            "by_k_input": by_k_input,
            "by_k_total_with_generation": by_k_total,
            f"k{K_WORST}_input": n32_in,
            f"k{K_WORST}_total_with_generation": n32_tot,
            "headroom_tokens_at_k_worst": window - n32_tot,
            "fits_effective_window_at_k_worst": fits_win,
            "fits_16k_token_count_at_k_worst": fits_16k,
        }
        report["models"][hf_id] = entry
        print(f"[ok] {hf_id} k={K_WORST} total={n32_tot} janela_efetiva={window} cabe={fits_win}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Atalho estável para o plot
    latest = out_dir / "latest.json"
    latest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nJSON: {out_path}\nAtalho: {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
