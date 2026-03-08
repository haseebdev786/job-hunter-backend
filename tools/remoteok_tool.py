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
            str(job.get("position", "")),
            str(job.get("description", "")),
            " ".join(str(tag) for tag in (job.get("tags", []) or [])),
        ]
    ).lower()
    return sum(1 for term in terms if term in haystack)


async def search_remoteok(query: str = "") -> List[dict]:
    """
    RemoteOK public API - free and no key required.
    Note: First list item is metadata/legal info and is skipped.
    """
    async with httpx.AsyncClient(
        timeout=30,
        headers={"User-Agent": "JobHunterAgent/1.0"},
    ) as client:
        try:
            response = await client.get("https://remoteok.com/api")
            response.raise_for_status()
            data = response.json()
            jobs = [j for j in data[1:] if isinstance(j, dict) and j.get("position")]

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
                    "job_id": str(j.get("id", "")),
                    "title": j.get("position", ""),
                    "company": j.get("company", ""),
                    "location": "Remote",
                    "description": j.get("description", "")[:2000],
                    "apply_url": j.get("url", ""),
                    "source": "remoteok",
                    "employment_type": "FULLTIME",
                }
                for j in jobs[:15]
            ]
        except Exception as exc:
            print(f"RemoteOK error: {exc}")
            return []
