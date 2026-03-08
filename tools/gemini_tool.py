from __future__ import annotations

import json
import re
from typing import Any

import httpx

from config import settings

_RESOLVED_MODEL: str | None = None
_DISCOVERED_MODELS: list[str] | None = None


def gemini_enabled() -> bool:
    return bool((settings.gemini_api_key or "").strip())


def _extract_text_from_response(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    first = candidates[0] or {}
    content = first.get("content") or {}
    parts = content.get("parts") or []
    texts: list[str] = []
    for part in parts:
        text = str((part or {}).get("text") or "").strip()
        if text:
            texts.append(text)
    return "\n".join(texts).strip()


def _safe_json_loads(raw: str, default: Any) -> Any:
    text = (raw or "").strip()
    if not text:
        return default

    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        if not match:
            return default
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return default


def _clean_model_name(model_name: str) -> str:
    name = (model_name or "").strip()
    if name.startswith("models/"):
        return name.split("/", 1)[1]
    return name


# Substrings that indicate a model does NOT support plain TEXT output
_NON_TEXT_MARKERS = (
    "tts", "image-generation", "imagen", "embedding",
    "aqa", "bisheng", "codec", "exp-image",
)


def _is_text_capable_model(name: str) -> bool:
    """Return True only if the model name looks like a normal text model."""
    lowered = name.lower()
    return not any(marker in lowered for marker in _NON_TEXT_MARKERS)


def _candidate_models(preferred: str | None = None) -> list[str]:
    base = [
        preferred or "",
        "gemini-1.5-flash",
    ]
    out: list[str] = []
    seen: set[str] = set()
    for item in base:
        cleaned = _clean_model_name(item)
        if not cleaned or cleaned in seen:
            continue
        if not _is_text_capable_model(cleaned):
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


async def _discover_models(client: httpx.AsyncClient, api_key: str) -> list[str]:
    global _DISCOVERED_MODELS
    if _DISCOVERED_MODELS is not None:
        return _DISCOVERED_MODELS

    discovered: list[str] = []
    for url in (
        "https://generativelanguage.googleapis.com/v1beta/models",
        "https://generativelanguage.googleapis.com/v1/models",
    ):
        try:
            response = await client.get(url, params={"key": api_key})
            if response.status_code >= 400:
                continue
            payload = response.json()
            for item in payload.get("models", []) or []:
                methods = item.get("supportedGenerationMethods") or []
                if "generateContent" not in methods:
                    continue
                name = _clean_model_name(str(item.get("name", "")))
                if name and _is_text_capable_model(name):
                    discovered.append(name)
            if discovered:
                break
        except Exception:
            continue

    # keep stable order + uniqueness
    unique: list[str] = []
    seen: set[str] = set()
    for name in discovered:
        if name in seen:
            continue
        seen.add(name)
        unique.append(name)

    _DISCOVERED_MODELS = unique
    return unique


def _ordered_models(preferred: str | None, discovered: list[str]) -> list[str]:
    preferred_list = _candidate_models(preferred)
    if not discovered:
        return preferred_list

    order: list[str] = []
    seen: set[str] = set()

    # first try preferred aliases if present in discovered
    for name in preferred_list:
        if name in discovered and name not in seen:
            seen.add(name)
            order.append(name)

    # then try any discovered models
    for name in discovered:
        if name not in seen:
            seen.add(name)
            order.append(name)

    # finally keep fallback names not discovered (in case list endpoint unavailable/stale)
    for name in preferred_list:
        if name not in seen:
            seen.add(name)
            order.append(name)

    return order


async def gemini_generate_text(
    prompt: str,
    *,
    system_prompt: str | None = None,
    history: list[dict[str, str]] | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 512,
    model: str | None = None,
) -> str:
    global _RESOLVED_MODEL
    api_key = (settings.gemini_api_key or "").strip()
    if not api_key:
        return ""

    preferred_model = _clean_model_name(model or _RESOLVED_MODEL or settings.gemini_model)

    contents: list[dict[str, Any]] = []
    for msg in history or []:
        role = "model" if str(msg.get("role", "")).lower() == "assistant" else "user"
        text = str(msg.get("content") or msg.get("text") or "").strip()
        if not text:
            continue
        contents.append({"role": role, "parts": [{"text": text[:12000]}]})

    contents.append({"role": "user", "parts": [{"text": prompt[:24000]}]})

    body: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max(32, min(max_output_tokens, 8192)),
        },
    }
    if system_prompt:
        body["systemInstruction"] = {"parts": [{"text": system_prompt[:8000]}]}

    async with httpx.AsyncClient(timeout=90) as client:
        discovered = await _discover_models(client, api_key)
        models_to_try = _ordered_models(preferred_model, discovered)

        last_error: str = ""
        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            try:
                response = await client.post(url, params={"key": api_key}, json=body)
                if response.status_code == 404:
                    last_error = f"model_not_found:{model_name}"
                    continue
                if response.status_code >= 400:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    if response.status_code in (400, 401, 403):
                        break  # Stop fallbacks on bad requests / auth errors
                    continue
                response.raise_for_status()
                _RESOLVED_MODEL = model_name
                return _extract_text_from_response(response.json())
            except httpx.RequestError as exc:
                last_error = f"RequestError: {exc}"
                continue
            except Exception as exc:
                last_error = f"Error: {exc}"
                continue

        if models_to_try:
            print(f"Gemini request warning: no working model. tried={models_to_try[:6]} last_error={last_error}")
        else:
            print("Gemini request warning: no model candidates available.")
        return ""


async def gemini_generate_json(
    prompt: str,
    default: Any,
    *,
    system_prompt: str | None = None,
    history: list[dict[str, str]] | None = None,
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
    model: str | None = None,
) -> Any:
    text = await gemini_generate_text(
        prompt,
        system_prompt=system_prompt,
        history=history,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        model=model,
    )
    parsed = _safe_json_loads(text, default)
    if isinstance(default, dict) and not isinstance(parsed, dict):
        return default
    if isinstance(default, list) and not isinstance(parsed, list):
        return default
    return parsed
