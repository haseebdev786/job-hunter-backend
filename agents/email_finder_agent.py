from __future__ import annotations

import asyncio
import re
from typing import Any, Callable

from tools.gemini_tool import gemini_enabled, gemini_generate_text
from tools.email_finder_tool import find_hr_email


async def _call_llm_text(prompt: str, default: str = "") -> str:
    if not gemini_enabled():
        return default

    try:
        text = await gemini_generate_text(
            prompt,
            temperature=0,
            max_output_tokens=120,
        )
        return text.split()[0].strip().lower().replace("http://", "").replace("https://", "").replace("www.", "")
    except Exception:
        return default


# Common business suffixes to strip when generating domain variations
_BIZ_SUFFIXES = [
    "gmbh", "inc", "ltd", "llc", "corp", "corporation", "co",
    "ag", "sa", "srl", "pvt", "private", "limited", "group",
    "holding", "holdings", "technologies", "technology", "tech",
    "solutions", "services", "software", "labs", "studio", "studios",
]

# TLDs to try for domain variations
_TLDS = [".com", ".io", ".de", ".tech", ".co", ".ai", ".org", ".net"]


def _generate_domain_candidates(company_name: str, guessed_domain: str = "") -> list[str]:
    """
    Generate multiple domain candidates from company name.
    E.g., "Ecoturn GmbH" -> ["ecoturn.com", "ecoturn.de", "ecoturn.io", "ecoturngmbh.com"]
    """
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(domain: str) -> None:
        d = domain.strip().lower()
        if d and d not in seen and "." in d:
            seen.add(d)
            candidates.append(d)

    # 1. LLM-guessed domain first (highest priority)
    if guessed_domain:
        _add(guessed_domain)

    # 2. Clean company name - strip suffixes and generate variations
    name = company_name.lower().strip()
    for suffix in _BIZ_SUFFIXES:
        pattern = rf"\b{re.escape(suffix)}\b"
        name = re.sub(pattern, "", name)
    clean = re.sub(r"[^a-z0-9]", "", name).strip()

    if clean:
        for tld in _TLDS:
            _add(f"{clean}{tld}")

    # 3. Also try with spaces as hyphens (e.g., "Judi Health" -> "judi-health.com")
    hyphenated = re.sub(r"[^a-z0-9]+", "-", company_name.lower().strip()).strip("-")
    for suffix in _BIZ_SUFFIXES:
        hyphenated = re.sub(rf"-?{re.escape(suffix)}-?", "-", hyphenated).strip("-")
    if hyphenated and hyphenated != clean:
        _add(f"{hyphenated}.com")

    # 4. Full name without cleanup (e.g., "ecoturngmbh.com")
    full = re.sub(r"[^a-z0-9]", "", company_name.lower())
    if full and full != clean:
        _add(f"{full}.com")

    return candidates


async def run_email_finder_agent(state: dict[str, Any], emit: Callable) -> dict[str, Any]:
    """
    For each job:
    1. Ask Gemini to guess company domain from company name
    2. Generate multiple domain variations
    3. Try all strategies: Gemini AI, HTTP scraping, Snov.io, Prospeo
    4. Final fallback is pattern matching

    Emits retry/progress events so UI can keep users informed.
    """
    jobs = state.get("jobs", [])
    found: list[dict] = []

    for job in jobs:
        company = job.get("company", "")
        await emit("email_finder", "FINDING_EMAIL", {"company": company}, "running")

        prompt = (
            f"What is the official website domain for the company \"{company}\"?\n"
            "Return ONLY the domain like: google.com\n"
            "No http, no www, just the domain. Be precise, double-check spelling."
        )
        guessed = await _call_llm_text(prompt, default="")
        guessed = guessed.strip().strip("/\n\r\t ").split("/")[0]

        # Two attempts: first fast/narrow, second wider.
        all_domain_candidates = _generate_domain_candidates(company, guessed)[:5]
        if not all_domain_candidates:
            all_domain_candidates = [guessed] if guessed else []

        attempt_candidates = [
            all_domain_candidates[:2] if all_domain_candidates else [],
            all_domain_candidates,
        ]
        max_attempts = 2
        attempts_used = 0
        result: dict[str, Any] = {
            "email": None,
            "confidence": 0.0,
            "source": "not_found",
            "provider_reason": "provider_failed",
            "provider_path": [],
        }

        for attempt_no, domain_candidates in enumerate(attempt_candidates, start=1):
            if not domain_candidates:
                continue

            attempts_used = attempt_no
            await emit(
                "email_finder",
                "TRYING_DOMAINS",
                {
                    "company": company,
                    "attempt": attempt_no,
                    "max_attempts": max_attempts,
                    "domains": domain_candidates[:5],
                },
                "running",
            )

            try:
                result = await asyncio.wait_for(
                    find_hr_email(
                        company,
                        domain=guessed or None,
                        domain_candidates=domain_candidates,
                    ),
                    timeout=90,
                )
            except asyncio.TimeoutError:
                result = {
                    "email": None,
                    "confidence": 0.0,
                    "source": "timeout",
                    "provider_reason": "email_search_timed_out_90s",
                    "provider_path": ["timeout"],
                }
                await emit(
                    "email_finder",
                    "EMAIL_TIMEOUT",
                    {
                        "company": company,
                        "attempt": attempt_no,
                        "max_attempts": max_attempts,
                        "message": "Email search attempt timed out after 90 seconds",
                    },
                    "running",
                )

            if result.get("email"):
                break

            if attempt_no < max_attempts:
                await emit(
                    "email_finder",
                    "EMAIL_RETRYING",
                    {
                        "company": company,
                        "attempt": attempt_no,
                        "next_attempt": attempt_no + 1,
                        "max_attempts": max_attempts,
                        "reason": str(result.get("provider_reason", "first_attempt_failed")),
                    },
                    "running",
                )

        if result.get("email"):
            provider_reason = str(result.get("provider_reason", "")).strip()
            provider_path = result.get("provider_path", [])
            payload = {
                "job_id": job.get("job_id", ""),
                "company": company,
                "domain": guessed,
                "email": result.get("email"),
                "confidence": float(result.get("confidence", 0.0)),
                "source": result.get("source", "unknown"),
                "provider_reason": provider_reason,
                "provider_path": provider_path,
            }
            found.append(payload)

            if provider_reason and "switched_to_prospeo" in provider_reason:
                await emit(
                    "email_finder",
                    "PROVIDER_SWITCH",
                    {
                        "company": company,
                        "domain": guessed,
                        "message": provider_reason,
                    },
                    "running",
                )

            await emit(
                "email_finder",
                "EMAIL_FOUND",
                {
                    "company": company,
                    "email": payload["email"],
                    "confidence": payload["confidence"],
                    "source": payload["source"],
                    "provider_reason": provider_reason,
                    "provider_path": provider_path,
                    "attempts_used": attempts_used,
                    "message": provider_reason or f"resolved_via_{payload['source']}",
                },
                "done",
            )
        else:
            provider_reason = str(result.get("provider_reason", "provider_failed"))
            provider_path = result.get("provider_path", [])
            await emit(
                "email_finder",
                "EMAIL_NOT_FOUND",
                {
                    "company": company,
                    "domain": guessed,
                    "domains_tried": all_domain_candidates[:5],
                    "attempts_used": attempts_used,
                    "provider_reason": provider_reason,
                    "provider_path": provider_path,
                    "tools_tried": len(provider_path),
                    "message": f"Tried {len(provider_path)} tools: {' -> '.join(provider_path[:7])}. {provider_reason}",
                },
                "running",
            )

    state["hr_emails"] = found
    return state
