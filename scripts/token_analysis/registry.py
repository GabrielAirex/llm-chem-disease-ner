"""Registry a partir dos scripts vLLM (`MODEL=`, `--max-model-len`)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# HF id no script vLLM → repo do tokenizer quando difere
TOKENIZER_ALIASES: Dict[str, str] = {
    "mistralai/Mistral-Nemo-12B-Instruct-v1": "mistralai/Mistral-Nemo-Instruct-2407",
    "meta-llama/Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
}

TARGET_16K = 16384

# Se algum .sh ainda tiver max-model-len conservador, força o valor usado na dissertação.
VLLM_CONTEXT_MAX_OVERRIDES: Dict[str, int] = {
    "meta-llama/Llama-3.2-3B-Instruct": 16384,
}


def apply_context_max_override(hf_id: str, info: Dict[str, Any]) -> Dict[str, Any]:
    """Cópia de `info` com max_model_len_max substituído quando há override explícito."""
    out = dict(info)
    if hf_id in VLLM_CONTEXT_MAX_OVERRIDES:
        forced = VLLM_CONTEXT_MAX_OVERRIDES[hf_id]
        orig = int(out.get("max_model_len_max", forced))
        if orig != forced:
            out["max_model_len_max_from_script"] = orig
        out["max_model_len_max"] = forced
    return out


# Tabela da dissertação (ordem das linhas LaTeX → id Hugging Face nos scripts vLLM)
DISSERTATION_MODEL_HF_IDS: List[str] = [
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "01-ai/Yi-1.5-9B-Chat",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "aaditya/Llama3-OpenBioLLM-8B",
    "internlm/internlm2_5-1_8b-chat",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "internlm/internlm2_5-7b-chat",
    "h2oai/h2o-danube3-4b-chat",
    "microsoft/Phi-3-mini-128k-instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "google/gemma-1.1-2b-it",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "Qwen/Qwen2-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
]

# Rótulos curtos iguais à tese (para eixos do plot)
DISSERTATION_LABELS: Dict[str, str] = {
    "meta-llama/Meta-Llama-3.1-70B-Instruct": "Llama-3.1-70B",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5-14B",
    "01-ai/Yi-1.5-9B-Chat": "Yi-1.5-9B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "aaditya/Llama3-OpenBioLLM-8B": "OpenBioLLM-8B",
    "internlm/internlm2_5-1_8b-chat": "InternLM2.5-1.8B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B-v0.3",
    "internlm/internlm2_5-7b-chat": "InternLM2.5-7B",
    "h2oai/h2o-danube3-4b-chat": "H2O-Danube3-4B",
    "microsoft/Phi-3-mini-128k-instruct": "Phi-3-128k",
    "microsoft/Phi-3-mini-4k-instruct": "Phi-3-4k",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B",
    "google/gemma-1.1-2b-it": "Gemma-1.1-2B",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "SmolLM2-1.7B",
    "Qwen/Qwen2-1.5B-Instruct": "Qwen2-1.5B",
    "meta-llama/Llama-3.2-1B-Instruct": "Llama-3.2-1B",
}


def synthetic_registry_entry(hf_id: str) -> Dict[str, Any]:
    """Quando o modelo não aparece em nenhum .sh: assume 16k (só para janela efetiva)."""
    return {
        "hf_id": hf_id,
        "max_model_len_min": TARGET_16K,
        "max_model_len_max": TARGET_16K,
        "scripts": [],
    }


def parse_vllm_registry(vllm_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []

    for path in sorted(vllm_dir.glob("*.sh")):
        if path.name in ("build_sif.sh",):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        m_model = re.search(r"^MODEL=(.+)$", text, re.MULTILINE)
        if not m_model:
            continue
        raw = m_model.group(1).strip().strip('"').strip("'")
        m_len = re.search(r"--max-model-len\s+(\d+)", text)
        if not m_len:
            m_len = re.search(r"max-model-len\s+(\d+)", text)
        if not m_len:
            continue
        mlen = int(m_len.group(1))
        if raw not in by_id:
            by_id[raw] = {
                "hf_id": raw,
                "max_model_len_min": mlen,
                "max_model_len_max": mlen,
                "scripts": [path.name],
            }
        else:
            by_id[raw]["scripts"].append(path.name)
            by_id[raw]["max_model_len_min"] = min(by_id[raw]["max_model_len_min"], mlen)
            by_id[raw]["max_model_len_max"] = max(by_id[raw]["max_model_len_max"], mlen)

    for hf_id, info in by_id.items():
        if info["max_model_len_min"] != info["max_model_len_max"]:
            warnings.append(
                f"{hf_id}: max-model-len varia entre scripts "
                f"({info['max_model_len_min']}–{info['max_model_len_max']})."
            )

    return by_id, warnings


def tokenizer_repo_id(hf_id: str) -> str:
    return TOKENIZER_ALIASES.get(hf_id, hf_id)


def effective_context_window(info: Dict[str, Any]) -> int:
    """min(16k, melhor max-model-len configurado nos .sh)."""
    mmax = int(info.get("max_model_len_max", TARGET_16K))
    return min(TARGET_16K, mmax)
