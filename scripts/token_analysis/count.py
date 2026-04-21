"""Contagem de tokens com chat template (alinhado a vLLM / OpenAI)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

TEXT_MARKER_FEW_SHOT = "\n\nNow analyze this text:\n\n"


def split_type2_few_shot(prompt: str) -> Tuple[Optional[str], str]:
    if TEXT_MARKER_FEW_SHOT not in prompt:
        return None, prompt
    parts = prompt.split(TEXT_MARKER_FEW_SHOT, 1)
    if len(parts) != 2:
        return None, prompt
    return parts[0].strip(), parts[1].strip()


def count_tokens_with_template(
    tokenizer: Any,
    system_prompt: Optional[str],
    user_content: str,
) -> int:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    if getattr(tokenizer, "chat_template", None):
        try:
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors=None,
            )
            return len(ids)
        except Exception:
            pass

    joined = (system_prompt + "\n\n" if system_prompt else "") + user_content
    return len(tokenizer.encode(joined, add_special_tokens=False))


def count_prompt(
    tokenizer: Any,
    prompt: str,
    use_chat_separation: bool,
) -> int:
    if use_chat_separation:
        sys_p, usr = split_type2_few_shot(prompt)
        return count_tokens_with_template(tokenizer, sys_p, usr)
    return count_tokens_with_template(tokenizer, None, prompt)
