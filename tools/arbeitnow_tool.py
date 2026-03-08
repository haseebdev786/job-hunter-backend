import re
from typing import List

import httpx

STOPWORDS = {
    "and",
    "the",
    "for",
    "with",
    "from",
    "this",
    "that",
    "have",
    "has",
    "your",
    "remote",
}


def _query_terms(query: str) -> list[str]:
    terms = re.findall(r"[a-z0-9+#]{2,}", (query or "").lower())
    return [term for term in terms if term not in STOPWORDS][:6]


def _score_job(job: dict, terms: list[str]) -> int:
    haystack = " ".join(
        [
            str(job.get("title", "")),
            str(job.get("description", "")),
            " ".join(str(tag) for tag in (job.get("tags", []) or [])),
        ]
    ).lower()
    return sum(1 for term in terms if term in haystack)


async def search_arbeitnow(query: str = "") -> List[dict]:
    """
    Arbeitnow public API - free and no key required.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.get(
                "https://www.arbeitnow.com/api/job-board-api",
                params={"page": 1},
            )
            response.raise_for_status()
            jobs = response.json().get("data", [])

            if query:
                terms = _query_terms(query)
                if terms:
                    scored = [(_score_job(job, terms), job) for job in jobs]
                    scored = [item for item in scored if item[0] > 0]
                    if scored:
                        scored.sort(key=lambda item: item[0], reverse=True)
                        jobs = [job for _, job in scored]

            return [
                {
                    "job_id": j.get("slug", ""),
                    "title": j.get("title", ""),
                    "company": j.get("company_name", ""),
                    "location": j.get("location", "Remote"),
                    "description": j.get("description", "")[:2000],
                    "apply_url": j.get("url", ""),
                    "source": "arbeitnow",
                    "employment_type": "FULLTIME",
                }
                for j in jobs[:20]
            ]
        except Exception as exc:
            print(f"Arbeitnow error: {exc}")
            return []
