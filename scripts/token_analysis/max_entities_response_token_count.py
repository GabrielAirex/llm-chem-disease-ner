#!/usr/bin/env python3
"""
Percorre indicios_encontrados/*/type2/**/extraction_*.json (pastas da dissertação),
reconstrói o JSON de saída no formato do prompt type2 (`{"entities":[...]}`),
encontra a **maior** resposta (por comprimento UTF-8 desse JSON minificado)
e conta tokens desse mesmo texto com **cada tokenizer** dos 18 modelos da dissertação.

Saída: reports/token_analysis/max_entities_response_tokens.json (+ impressão na consola)

Uso (raiz onconavegation-BioNER_llm):

  python3 scripts/token_analysis/max_entities_response_token_count.py
  HF_TOKEN=hf_... python3 scripts/token_analysis/max_entities_response_token_count.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

# Pasta slug (indicios_encontrados) → HF id (dissertação)
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


def entities_from_extraction(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Normaliza chemicals/diseases ou lista plana → lista {text, type}."""
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
    """Mesmo formato lógico que o prompt pede (lista entities)."""
    return json.dumps({"entities": entities}, ensure_ascii=False, separators=(",", ":"))


def scan_indicios(
    base: Path,
) -> Tuple[str, int, str, str, int, List[Dict[str, str]]]:
    """
    Retorna: path_vencedor, num_bytes_utf8, slug_modelo, pmid, num_entidades, entities_list
    """
    best_path = ""
    best_len = -1
    best_slug = ""
    best_pmid = ""
    best_entities: List[Dict[str, str]] = []
    n_files = 0

    for path in sorted(base.rglob("extraction_*.json")):
        parts = path.parts
        if "type2" not in parts:
            continue
        slug = None
        try:
            idx = parts.index("indicios_encontrados")
            slug = parts[idx + 1]
        except (ValueError, IndexError):
            for i, p in enumerate(parts):
                if p in SLUG_TO_HF:
                    slug = p
                    break
        if slug is None or slug not in SLUG_TO_HF:
            continue

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        n_files += 1
        entities = entities_from_extraction(data)
        text = canonical_response_json(entities)
        L = len(text.encode("utf-8"))
        if L > best_len:
            best_len = L
            best_path = str(path)
            best_slug = slug
            best_pmid = str(data.get("pmid", ""))
            best_entities = entities

    return best_path, best_len, best_slug, best_pmid, n_files, best_entities


def count_tokens_raw(tokenizer: Any, text: str) -> int:
    """Só o texto da geração (JSON), sem chat template."""
    return len(tokenizer.encode(text, add_special_tokens=False))


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
        default=ROOT / "reports" / "token_analysis" / "max_entities_response_tokens.json",
    )
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--local-only", action="store_true")
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

    print("A varrer extraction_*.json em */type2/ ...")
    best_path, best_len, best_slug, best_pmid, n_files, best_entities = scan_indicios(
        args.indicios_dir
    )
    if not best_path:
        print("Nenhum ficheiro válido encontrado.")
        return 1

    response_json = canonical_response_json(best_entities)
    n_ent = len(best_entities)

    print(f"  Ficheiros considerados: {n_files}")
    print(f"  Maior JSON (UTF-8 bytes): {best_len}")
    print(f"  Entidades: {n_ent}")
    print(f"  pmid: {best_pmid}  slug: {best_slug}")
    print(f"  Ficheiro: {best_path}")
    print(f"  Pré-visualização: {response_json[:200]}{'...' if len(response_json) > 200 else ''}")
    print()

    results: Dict[str, Any] = {
        "winner": {
            "path": best_path,
            "pmid": best_pmid,
            "slug": best_slug,
            "hf_id_folder": SLUG_TO_HF.get(best_slug),
            "num_entities": n_ent,
            "response_json_utf8_bytes": best_len,
            "response_json": response_json,
        },
        "files_scanned": n_files,
        "token_counts_by_model": {},
        "errors": {},
    }

    for hf_id in DISSERTATION_MODEL_HF_IDS:
        label = DISSERTATION_LABELS.get(hf_id, hf_id)
        tok_id = tokenizer_repo_id(hf_id)
        try:
            tok_kw: Dict[str, Any] = {
                "trust_remote_code": True,
                "local_files_only": args.local_only,
                "token": hf_token,
            }
            if hf_id.startswith("internlm/"):
                tok_kw["use_fast"] = False
            tok = AutoTokenizer.from_pretrained(tok_id, **tok_kw)
            n_tok = count_tokens_raw(tok, response_json)
            results["token_counts_by_model"][hf_id] = {
                "label_short": label,
                "tokenizer_hf_id": tok_id,
                "num_tokens": n_tok,
            }
            print(f"  {label:22}  {tok_id:48}  {n_tok:5} tokens")
        except Exception as e:
            results["errors"][hf_id] = str(e)
            print(f"  {label:22}  ERRO: {e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Não repetir JSON gigante duas vezes no ficheiro se for enorme
    out_copy = json.loads(json.dumps(results))
    if len(response_json) > 8000:
        out_copy["winner"]["response_json"] = response_json[:4000] + "\n... [truncado no JSON de relatório; campo completo em winner.response_json_full abaixo]"
        out_copy["winner"]["response_json_full"] = response_json
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_copy, f, indent=2, ensure_ascii=False)

    print(f"\nJSON: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
