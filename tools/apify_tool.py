import asyncio
import html
import re
from typing import List

import httpx

try:
    from apify_client import ApifyClient
except Exception:  # pragma: no cover - optional dependency guard
    ApifyClient = None  # type: ignore[assignment]

from config import settings

_LAST_LINKEDIN_PROVIDER_REASON = "init"


def apify_linkedin_enabled() -> bool:
    return bool(settings.apify_enabled and settings.apify_api_token and ApifyClient is not None)


def get_linkedin_provider_reason() -> str:
    return _LAST_LINKEDIN_PROVIDER_REASON


def _set_provider_reason(reason: str) -> None:
    global _LAST_LINKEDIN_PROVIDER_REASON
    _LAST_LINKEDIN_PROVIDER_REASON = reason


def _is_quiet_apify_error(exc: Exception) -> bool:
    message = str(exc).lower()
    quiet_markers = [
        "must rent a paid actor",
        "free trial has expired",
        "quota",
        "insufficient",
        "credits",
        "unauthorized",
        "forbidden",
    ]
    return any(marker in message for marker in quiet_markers)


def _map_apify_error(exc: Exception) -> str:
    message = str(exc).lower()
    if "must rent a paid actor" in message or "free trial has expired" in message:
        return "apify_paid_actor_required"
    if "quota" in message or "credits" in message or "insufficient" in message:
        return "apify_quota_or_credits_exhausted"
    if "unauthorized" in message or "forbidden" in message:
        return "apify_auth_error"
    return "apify_request_error"


def _strip_html(value: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", value or "", flags=re.S))
    return re.sub(r"\s+", " ", text).strip()


def _parse_guest_cards(payload: str, default_location: str) -> List[dict]:
    cards = re.findall(r"<li[^>]*>.*?</li>", payload or "", flags=re.S | re.I)
    jobs: list[dict] = []
    for card in cards:
        if "base-search-card" not in card and "base-card__full-link" not in card:
            continue

        job_id_match = re.search(r"urn:li:jobPosting:(\d+)", card, flags=re.I)
        job_id = job_id_match.group(1) if job_id_match else ""

        title_match = re.search(r'base-search-card__title[^>]*>(.*?)</h3>', card, flags=re.S | re.I)
        company_match = re.search(r'base-search-card__subtitle[^>]*>.*?<a[^>]*>(.*?)</a>', card, flags=re.S | re.I)
        if not company_match:
            company_match = re.search(r'base-search-card__subtitle[^>]*>(.*?)</h4>', card, flags=re.S | re.I)
        location_match = re.search(r'job-search-card__location[^>]*>(.*?)</span>', card, flags=re.S | re.I)
        url_match = re.search(r'base-card__full-link[^>]*href="([^"]+)"', card, flags=re.S | re.I)

        title = _strip_html(title_match.group(1) if title_match else "")
        company = _strip_html(company_match.group(1) if company_match else "")
        location = _strip_html(location_match.group(1) if location_match else "") or default_location
        apply_url = html.unescape(url_match.group(1)) if url_match else ""

        if not title:
            continue

        jobs.append(
            {
                "job_id": job_id,
                "title": title,
                "company": company,
                "location": location,
                "description": "",
                "apply_url": apply_url,
                "source": "linkedin_guest",
                "employment_type": "FULLTIME",
            }
        )
    return jobs


def _extract_guest_description(payload: str) -> str:
    if not payload:
        return ""
    match = re.search(
        r'show-more-less-html__markup[^>]*>(.*?)</div>',
        payload,
        flags=re.S | re.I,
    )
    if not match:
        return ""
    return _strip_html(match.group(1))


async def _search_linkedin_guest(
    keywords: str,
    location: str,
    max_results: int,
) -> List[dict]:
    _set_provider_reason("linkedin_guest_search")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    collected: list[dict] = []
    seen: set[str] = set()

    async with httpx.AsyncClient(timeout=30, headers=headers, follow_redirects=True) as client:
        for start in range(0, max(25, max_results * 3), 25):
            if len(collected) >= max_results:
                break
            try:
                response = await client.get(
                    "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search",
                    params={
                        "keywords": keywords,
                        "location": location,
                        "start": start,
                    },
                )
            except Exception:
                _set_provider_reason("linkedin_guest_request_error")
                break

            if response.status_code >= 400:
                _set_provider_reason(f"linkedin_guest_http_{response.status_code}")
                break

            cards = _parse_guest_cards(response.text, location)
            if not cards:
                break

            for job in cards:
                strong_id = str(job.get("job_id") or "").strip()
                key = strong_id or f"{job.get('title', '')}:{job.get('company', '')}:{job.get('location', '')}"
                if key in seen:
                    continue
                seen.add(key)
                collected.append(job)
                if len(collected) >= max_results:
                    break

        for job in collected[: min(len(collected), 8)]:
            job_id = str(job.get("job_id") or "").strip()
            if not job_id:
                continue
            try:
                detail_resp = await client.get(
                    f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
                )
                if detail_resp.status_code < 400:
                    description = _extract_guest_description(detail_resp.text)[:2000]
                    if description:
                        job["description"] = description
            except Exception:
                continue

    if collected:
        _set_provider_reason("linkedin_guest_ok")
    else:
        _set_provider_reason("linkedin_guest_empty")
    return collected[:max_results]


def _search_linkedin_apify_sync(keywords: str, location: str, max_results: int) -> List[dict]:
    if not apify_linkedin_enabled():
        _set_provider_reason("apify_disabled")
        return []

    client = ApifyClient(settings.apify_api_token)

    # Try free actors in order of reliability
    actors_to_try = [
        {
            "id": "worldunboxer/rapid-linkedin-scraper",
            "input": {
                "searchUrl": f"https://www.linkedin.com/jobs/search/?keywords={keywords}&location={location}&f_TPR=r604800",
                "maxItems": max_results,
            },
            "map": lambda item: {
                "job_id": str(item.get("id") or item.get("jobId") or ""),
                "title": item.get("title") or item.get("position") or "",
                "company": item.get("companyName") or item.get("company") or "",
                "location": item.get("location") or location,
                "description": (item.get("description") or item.get("descriptionText") or "")[:2000],
                "apply_url": item.get("applyUrl") or item.get("jobUrl") or item.get("link") or "",
                "source": "linkedin_apify",
                "employment_type": item.get("employmentType", "FULLTIME"),
            },
        },
        {
            "id": "BHzefUZlZRKWxkTck",
            "input": {
                "searchQueries": [{"keywords": keywords, "location": location, "dateSincePosted": "pastWeek"}],
                "maxResults": max_results,
                "includeJobDescriptions": True,
            },
            "map": lambda item: {
                "job_id": item.get("id", ""),
                "title": item.get("title", ""),
                "company": item.get("companyName", ""),
                "location": item.get("location", location),
                "description": item.get("descriptionText", "")[:2000],
                "apply_url": item.get("applyUrl") or item.get("jobUrl", ""),
                "source": "linkedin_apify",
                "employment_type": item.get("employmentType", "FULLTIME"),
            },
        },
    ]

    for actor_config in actors_to_try:
        try:
            run = client.actor(actor_config["id"]).call(run_input=actor_config["input"])
            jobs: List[dict] = []
            for item in client.dataset(run["defaultDatasetId"]).iterate_items():
                mapped = actor_config["map"](item)
                if mapped.get("title"):
                    jobs.append(mapped)
            if jobs:
                _set_provider_reason(f"apify_ok_via_{actor_config['id'].split('/')[-1]}")
                return jobs
        except Exception as exc:
            reason = _map_apify_error(exc)
            if not _is_quiet_apify_error(exc):
                print(f"Apify actor {actor_config['id']} warning: {exc}")
            if reason == "apify_paid_actor_required":
                continue  # try next actor
            _set_provider_reason(reason)
            return []

    _set_provider_reason("apify_all_actors_failed")
    return []


async def search_linkedin_jobs(
    keywords: str,
    location: str = "Remote",
    max_results: int = 10,
) -> List[dict]:
    if apify_linkedin_enabled():
        apify_jobs = await asyncio.to_thread(
            _search_linkedin_apify_sync,
            keywords,
            location,
            max_results,
        )
        if apify_jobs:
            return apify_jobs[:max_results]

    # Always fallback to free guest source when Apify is unavailable/empty/paid-blocked.
    guest_jobs = await _search_linkedin_guest(keywords, location, max_results)
    return guest_jobs[:max_results]
