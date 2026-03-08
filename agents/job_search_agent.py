from __future__ import annotations

import asyncio
import json
import re
from difflib import SequenceMatcher
from typing import Any, Callable

from tools.apify_tool import get_linkedin_provider_reason, search_linkedin_jobs
from tools.arbeitnow_tool import search_arbeitnow
from tools.remoteok_tool import search_remoteok
from tools.gemini_tool import gemini_enabled, gemini_generate_json

QUERY_STOPWORDS = {
    "and",
    "the",
    "for",
    "with",
    "from",
    "that",
    "this",
    "have",
    "has",
    "your",
    "such",
    "into",
    "about",
    "experience",
    "skills",
    "skill",
    "summary",
    "resume",
    "candidate",
}


async def _call_llm_json(prompt: str, default: Any) -> Any:
    if not gemini_enabled():
        return default

    try:
        return await gemini_generate_json(
            prompt,
            default=default,
            temperature=0.0,
            max_output_tokens=1000,
        )
    except Exception:
        return default


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", value.lower())).strip()


def _compact_query(raw_query: str, fallback: str = "software engineer") -> str:
    tokens = re.findall(r"[a-z0-9+#]{2,}", (raw_query or "").lower())
    filtered = [token for token in tokens if token not in QUERY_STOPWORDS][:6]

    if len(filtered) < 2:
        fallback_tokens = re.findall(r"[a-z0-9+#]{2,}", (fallback or "").lower())
        fallback_filtered = [token for token in fallback_tokens if token not in QUERY_STOPWORDS][:4]
        filtered = filtered + fallback_filtered

    if not filtered:
        return "software engineer"

    return " ".join(filtered[:4]).strip()


def _pick_three_queries(candidates: list[str], fallback: str) -> tuple[str, str, str]:
    unique_queries: list[str] = []
    seen: set[str] = set()

    defaults = [fallback, f"{fallback} remote", "software engineer remote"]
    for candidate in candidates + defaults:
        cleaned = _compact_query(str(candidate), fallback=fallback)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique_queries.append(cleaned)
        if len(unique_queries) == 3:
            break

    while len(unique_queries) < 3:
        unique_queries.append(_compact_query(fallback, fallback))

    return unique_queries[0], unique_queries[1], unique_queries[2]


def _dedupe_jobs(jobs: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen_ids: set[str] = set()

    for job in jobs:
        source = job.get("source", "")
        raw_id = str(job.get("job_id", "")).strip()
        strong_id = f"{source}:{raw_id}" if raw_id else ""
        if strong_id and strong_id in seen_ids:
            continue

        title = _normalize_text(job.get("title", ""))
        company = _normalize_text(job.get("company", ""))
        is_duplicate = False
        for existing in deduped:
            if company and company == _normalize_text(existing.get("company", "")):
                ratio = SequenceMatcher(a=title, b=_normalize_text(existing.get("title", ""))).ratio()
                if ratio >= 0.9:
                    is_duplicate = True
                    break

        if is_duplicate:
            continue

        if strong_id:
            seen_ids.add(strong_id)
        deduped.append(job)

    return deduped


def _fallback_queries(profile: dict, prefs: dict) -> list[str]:
    title = prefs.get("job_title") or "software engineer"
    skills = profile.get("skills") or []
    skill_1 = _compact_query(str(skills[0]), fallback="python") if len(skills) > 0 else "python"
    skill_2 = _compact_query(str(skills[1]), fallback="backend") if len(skills) > 1 else "backend"
    return [
        f"{title}",
        f"{skill_1} {title}",
        f"{skill_2} {title} remote",
    ]


# Competing framework groups — if job requires one and candidate has the other, it's a mismatch
_FRAMEWORK_GROUPS = [
    {"react", "next.js", "nextjs", "react.js", "reactjs"},
    {"angular", "angularjs", "angular.js"},
    {"vue", "vue.js", "vuejs", "nuxt", "nuxt.js", "nuxtjs"},
    {"django", "flask", "fastapi"},
    {"spring", "spring boot", "springboot"},
    {".net", "dotnet", "asp.net", "c#", "csharp"},
    {"ruby on rails", "rails", "ruby"},
    {"laravel", "php"},
]


def _detect_framework_mismatch(candidate_skills: list[str], job_text: str) -> bool:
    """Return True if the job requires a framework from a DIFFERENT group than what the candidate knows."""
    job_lower = job_text.lower()
    skills_lower = {s.lower() for s in candidate_skills}

    for group in _FRAMEWORK_GROUPS:
        job_has = any(fw in job_lower for fw in group)
        candidate_has = any(fw in skill for fw in group for skill in skills_lower)
        if job_has and not candidate_has:
            # Job requires this framework group, candidate doesn't have it
            # Check if candidate has a COMPETING framework group
            for other_group in _FRAMEWORK_GROUPS:
                if other_group is group:
                    continue
                if any(fw in skill for fw in other_group for skill in skills_lower):
                    return True  # Candidate has a competing framework
    return False


def _fallback_relevance(profile: dict, job: dict) -> dict:
    skills = [str(s).lower() for s in profile.get("skills", [])]
    text = f"{job.get('title', '')} {job.get('description', '')}".lower()
    hits = sum(1 for skill in skills if skill and skill in text)
    score = min(95, 35 + hits * 10)

    # Penalize framework mismatch heavily
    if _detect_framework_mismatch(skills, text):
        score = min(score, 15)
        return {"score": score, "reason": "Framework mismatch — job requires different tech stack"}

    return {"score": score, "reason": "Skill overlap heuristic"}


async def score_job_against_cv(job: dict, profile: dict, preferred_title: str) -> dict:
    """Scores a single job against a user profile using the LLM (or fallback heuristic)."""
    years = len(profile.get("experience", []) or [])
    skills = profile.get("skills", []) or []
    
    score_prompt = (
        "Rate job relevance for this candidate. Return ONLY JSON: {\"score\": 0-100, \"reason\": \"one sentence\"}\n\n"
        "CRITICAL RULES:\n"
        "- If the job REQUIRES a specific framework/language (e.g. Angular, Vue, Java, .NET) "
        "and the candidate's primary skills are in a DIFFERENT framework (e.g. React, Python), "
        "score MUST be BELOW 20. This is a hard mismatch.\n"
        "- 'Frontend Developer' jobs that specifically say Angular/Vue are NOT suitable for a React developer.\n"
        "- Only score above 60 if the candidate's core skills genuinely match the job requirements.\n"
        "- A job titled 'Angular Frontend Developer' for a React developer = score 10-15.\n\n"
        f"Candidate preferred role: {preferred_title}\n"
        f"Candidate: {years} years exp, skills: {skills}\n"
        f"Job: {job.get('title', '')} at {job.get('company', '')}\n"
        f"Description snippet: {job.get('description', '')[:400]}"
    )
    scoring = await _call_llm_json(score_prompt, _fallback_relevance(profile, job))
    if not isinstance(scoring, dict):
        scoring = _fallback_relevance(profile, job)
    return {
        **job,
        "relevance_score": float(scoring.get("score", 0)),
        "relevance_reason": str(scoring.get("reason", "")),
    }


async def run_job_search_agent(state: dict[str, Any], emit: Callable) -> dict[str, Any]:
    """
    1. Claude generates 3 search keyword strings from user profile
    2. Call LinkedIn (Apify) with generated queries
    3. Merge and deduplicate by id and title similarity
    4. Claude scores each job relevance 0-100
    5. Sort by score and return top jobs
    """
    profile = state.get("user_profile", {})
    prefs = state.get("preferences", {})
    location = prefs.get("location", "Remote")
    try:
        max_apps = int(prefs.get("max_applications", 10))
    except Exception:
        max_apps = 10
    max_apps = max(1, min(max_apps, 20))

    await emit("job_search", "GENERATING_QUERIES", {"status": "starting"}, "running")
    prompt = (
        "Based on this profile and preferences, generate 3 job search keyword strings.\n"
        f"Profile: {json.dumps(profile)}\n"
        f"Preferences: {json.dumps(prefs)}\n"
        "Return ONLY a JSON array of 3 strings, 2-4 words each.\n"
        "Example: [\"python backend developer\", \"django rest api engineer\", \"senior software engineer remote\"]"
    )
    queries = await _call_llm_json(prompt, _fallback_queries(profile, prefs))
    if not isinstance(queries, list) or len(queries) < 3:
        queries = _fallback_queries(profile, prefs)

    fallback_title = str(prefs.get("job_title") or "software engineer")
    q1, q2, q3 = _pick_three_queries([str(q) for q in queries[:3]], fallback_title)
    await emit(
        "job_search",
        "SEARCHING_LINKEDIN",
        {"queries": [q1, q2, q3], "source_mode": "linkedin_only"},
        "running",
    )

    results = await asyncio.gather(
        search_linkedin_jobs(q1, location=location, max_results=max(max_apps, 8)),
        search_linkedin_jobs(q2, location=location, max_results=max(max_apps, 8)),
        search_linkedin_jobs(q3, location=location, max_results=max(max_apps, 8)),
        return_exceptions=True,
    )

    source_counts: dict[str, int] = {}
    all_jobs: list[dict] = []
    for result in results:
        if isinstance(result, Exception):
            state.setdefault("errors", []).append(str(result))
            continue
        for job in result:
            source = str(job.get("source") or "linkedin_unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        all_jobs.extend(result)

    minimum_target = max(6, max_apps)
    if len(all_jobs) < minimum_target:
        broad_query = _compact_query(fallback_title, fallback="software engineer")
        await emit(
            "job_search",
            "BROADENING_SEARCH",
            {"reason": "initial_results_low", "query": broad_query, "initial_count": len(all_jobs), "source_mode": "linkedin_only"},
            "running",
        )
        try:
            extra = await search_linkedin_jobs(
                broad_query,
                location=location,
                max_results=max(max_apps * 2, 12),
            )
        except Exception as exc:  # pragma: no cover - defensive
            state.setdefault("errors", []).append(str(exc))
            extra = []
        for job in extra:
            source = str(job.get("source") or "linkedin_unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        all_jobs.extend(extra)

    # ── FALLBACK: If LinkedIn returned nothing, try free job boards ──
    if not all_jobs:
        await emit(
            "job_search",
            "FALLBACK_SOURCES",
            {"reason": "linkedin_empty", "sources": ["arbeitnow", "remoteok"]},
            "running",
        )
        fallback_results = await asyncio.gather(
            search_arbeitnow(fallback_title),
            search_remoteok(fallback_title),
            return_exceptions=True,
        )
        for result in fallback_results:
            if isinstance(result, Exception):
                state.setdefault("errors", []).append(f"Fallback source error: {result}")
                continue
            for job in result:
                source = str(job.get("source") or "fallback")
                source_counts[source] = source_counts.get(source, 0) + 1
            all_jobs.extend(result)

    await emit(
        "job_search",
        "SOURCE_SUMMARY",
        {
            **source_counts,
            "provider_reason": get_linkedin_provider_reason(),
        },
        "running",
    )

    jobs = _dedupe_jobs(all_jobs)
    for job in jobs:
        await emit(
            "job_search",
            "JOB_FOUND",
            {
                "job_id": job.get("job_id", ""),
                "title": job.get("title", ""),
                "company": job.get("company", ""),
                "source": job.get("source", ""),
                "apply_url": job.get("apply_url", ""),
            },
            "running",
        )

    await emit("job_search", "SCORING_RESULTS", {"count": len(jobs)}, "running")

    years = len(profile.get("experience", []) or [])
    skills = profile.get("skills", []) or []
    preferred_title = str(prefs.get("job_title", "")).strip()

    scored = await asyncio.gather(*(score_job_against_cv(job, profile, preferred_title) for job in jobs[:40])) if jobs else []
    # Filter out jobs with very low relevance (framework mismatches etc.)
    scored_filtered = [j for j in scored if j.get("relevance_score", 0) >= 25]
    if not scored_filtered and scored:
        # If all jobs scored below 25, keep the best ones anyway (at least 1)
        scored_filtered = sorted(scored, key=lambda x: x.get("relevance_score", 0), reverse=True)[:2]
    scored_sorted = sorted(scored_filtered, key=lambda x: x.get("relevance_score", 0), reverse=True)
    selected = scored_sorted[: max(1, min(max_apps, 15))]

    state["jobs"] = selected
    if not selected:
        provider_reason = get_linkedin_provider_reason()
        message = "No jobs found from any source for current filters."
        if provider_reason and provider_reason not in ("linkedin_guest_empty", "apify_empty"):
            message = f"{message} provider={provider_reason}"
        state.setdefault("errors", []).append(message)
        await emit(
            "job_search",
            "NO_JOBS_FOUND",
            {"message": message, "provider_reason": provider_reason, "location": location},
            "error",
        )

    sources_used = list(set(j.get("source", "") for j in selected)) if selected else []
    await emit("job_search", "SEARCH_COMPLETE", {"count_selected": len(selected), "sources": sources_used}, "done")
    return state
