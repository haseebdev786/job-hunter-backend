from __future__ import annotations

import asyncio
import re
from typing import Any, Optional

import httpx

try:
    import dns.resolver
except ImportError:
    dns = None  # type: ignore

try:
    from apify_client import ApifyClient
except Exception:
    ApifyClient = None  # type: ignore[assignment]

from config import settings
from tools.gemini_tool import gemini_enabled, gemini_generate_text

HR_PATTERNS = [
    "hr@{domain}",
    "careers@{domain}",
    "talent@{domain}",
    "jobs@{domain}",
    "recruiting@{domain}",
    "people@{domain}",
    "apply@{domain}",
    "hiring@{domain}",
    "info@{domain}",
    "contact@{domain}",
]

HR_KEYWORDS = ["hr", "talent", "recruit", "people", "career", "hiring", "human resources"]
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Skip these junk emails found on websites
_JUNK_EMAIL_MARKERS = (
    "@sentry", "@wixpress", "@example", "noreply",
    "no-reply", "@gravatar", "@googleapis", "@google.com",
    ".png", ".jpg", ".gif", ".svg", ".webp",
    "@w3.org", "@microsoft.com", "@apple.com",
    "schema.org", "@facebook.com", "@twitter.com",
    "privacy@", "abuse@", "postmaster@", "webmaster@",
    "support@", "admin@",
)


async def find_hr_email(company_name: str, domain: Optional[str] = None, domain_candidates: Optional[list[str]] = None) -> dict:
    """
    Multi-strategy email finder. Tries in order:
    1. Gemini AI (ask LLM to provide a known HR/contact email)
    2. Direct HTTP website scraping (company contact/careers pages)
    3. Snov.io API (if configured)
    4. Prospeo API (if configured)
    5. Smart pattern fallback (common HR email patterns)

    MX validation is done on the FOUND email, not upfront, so that
    Snov.io/Prospeo always get a chance to run even if the guessed
    domain is wrong.
    """
    # Build list of domains to try
    domains_to_try: list[str] = []
    if domain_candidates:
        for d in domain_candidates:
            nd = _normalize_domain(d)
            if nd and nd not in domains_to_try:
                domains_to_try.append(nd)
    if domain:
        nd = _normalize_domain(domain)
        if nd and nd not in domains_to_try:
            domains_to_try.insert(0, nd)

    if not domains_to_try:
        return {
            "email": None,
            "confidence": 0.0,
            "source": "not_found",
            "provider_reason": "invalid_domain",
            "provider_path": [],
        }

    # Try each domain candidate
    all_reasons: list[str] = []
    all_path: list[str] = []

    for normalized_domain in domains_to_try:
        provider_path: list[str] = []
        reasons: list[str] = []
        print(f"\n[EmailFinder] Trying domain: {normalized_domain} for company: {company_name}")

        # ── 1. Gemini AI: Ask the LLM for a real company email ──
        gemini_result, gemini_reason = await _search_gemini_email(company_name, normalized_domain)
        provider_path.append("gemini_ai")
        reasons.append(gemini_reason)
        print(f"  [1/7] Gemini AI: {gemini_reason} -> {gemini_result.get('email') if gemini_result else 'None'}")
        if gemini_result:
            validated = _validate_found_email(gemini_result, reasons, provider_path)
            if validated:
                return validated

        # ── 2. Direct HTTP scrape of company website ──
        scrape_result, scrape_reason = await _scrape_website_emails(normalized_domain, company_name)
        provider_path.append("http_scrape")
        reasons.append(scrape_reason)
        print(f"  [2/7] HTTP Scrape: {scrape_reason} -> {scrape_result.get('email') if scrape_result else 'None'}")
        if scrape_result:
            validated = _validate_found_email(scrape_result, reasons, provider_path)
            if validated:
                return validated

        # ── 3. Apify Contact Info Scraper ──
        apify_result, apify_reason = await _search_apify_email(normalized_domain)
        provider_path.append("apify_scraper")
        reasons.append(apify_reason)
        print(f"  [3/7] Apify Scraper: {apify_reason} -> {apify_result.get('email') if apify_result else 'None'}")
        if apify_result:
            validated = _validate_found_email(apify_result, reasons, provider_path)
            if validated:
                return validated

        # ── 4. Snov.io ──
        snov_result, snov_reason = await _search_snov(normalized_domain)
        provider_path.append("snov_io")
        reasons.append(snov_reason)
        print(f"  [4/7] Snov.io: {snov_reason} -> {snov_result.get('email') if snov_result else 'None'}")
        if snov_result:
            validated = _validate_found_email(snov_result, reasons, provider_path)
            if validated:
                return validated

        # ── 5. Prospeo Domain Search (simple, direct) ──
        prospeo_domain_result, prospeo_domain_reason = await _search_prospeo_domain(normalized_domain)
        provider_path.append("prospeo_domain")
        reasons.append(prospeo_domain_reason)
        print(f"  [5/7] Prospeo Domain: {prospeo_domain_reason} -> {prospeo_domain_result.get('email') if prospeo_domain_result else 'None'}")
        if prospeo_domain_result:
            validated = _validate_found_email(prospeo_domain_result, reasons, provider_path)
            if validated:
                return validated

        # ── 6. Prospeo Person Search + Enrich (fallback) ──
        prospeo_result, prospeo_reason = await _search_prospeo(normalized_domain)
        provider_path.append("prospeo_person")
        reasons.append(prospeo_reason)
        print(f"  [6/7] Prospeo Person: {prospeo_reason} -> {prospeo_result.get('email') if prospeo_result else 'None'}")
        if prospeo_result:
            validated = _validate_found_email(prospeo_result, reasons, provider_path)
            if validated:
                return validated

        # ── 7. Smart Pattern Fallback ──
        pattern_result, pattern_reason = await _smart_pattern_fallback(normalized_domain)
        provider_path.append("pattern_verified")
        reasons.append(pattern_reason)
        print(f"  [7/7] Pattern Fallback: {pattern_reason} -> {pattern_result.get('email') if pattern_result else 'None'}")
        if pattern_result:
            validated = _validate_found_email(pattern_result, reasons, provider_path)
            if validated:
                return validated

        all_reasons.extend(reasons)
        all_path.extend(provider_path)

    print(f"[EmailFinder] EXHAUSTED all sources for {company_name} across domains: {domains_to_try}")
    return {
        "email": None,
        "confidence": 0.0,
        "source": "not_found",
        "provider_reason": " -> ".join(all_reasons) + " -> exhausted_all_sources",
        "provider_path": all_path,
    }


def _validate_found_email(result: dict, reasons: list[str], provider_path: list[str]) -> Optional[dict]:
    """Validate a found email before accepting it.
    - Always check MX records
    - For GUESSED emails (gemini_ai, pattern), ALWAYS do SMTP verification
    - For found-from-data sources (snov, prospeo, scrape), only SMTP verify if low confidence
    """
    email = result.get("email", "")
    if not email or "@" not in email:
        return None

    email_domain = email.split("@")[1]
    if not verify_email_domain(email_domain):
        reasons.append(f"mx_failed:{email_domain}")
        print(f"Email {email} skipped: domain {email_domain} has no MX records")
        return None

    source = result.get("source", "")
    confidence = float(result.get("confidence", 0.0))

    # Sources that GUESS emails (not found from real data) — ALWAYS verify via SMTP
    guessing_sources = {"gemini_ai", "pattern_verified", "pattern_smtp_verified"}
    # Sources that actually FIND emails from real databases/APIs — trust more
    verified_sources = {"snov_io", "prospeo", "prospeo_domain", "website_scrape", "apify_scraper"}

    needs_smtp = (
        source in guessing_sources        # Guessed emails ALWAYS need SMTP
        or confidence < 0.7               # Low confidence from any source
        or source not in verified_sources  # Unknown sources need verification
    )

    if needs_smtp:
        smtp_ok = verify_email_smtp(email)
        if not smtp_ok:
            reasons.append(f"smtp_rejected:{email}")
            print(f"Email {email} skipped: SMTP verification failed (mailbox doesn't exist)")
            return None
        # Boost confidence if SMTP verified
        result["confidence"] = min(0.85, confidence + 0.2)
        result["smtp_verified"] = True
        print(f"Email {email} SMTP verified ✓")

    result["provider_reason"] = " -> ".join(reasons)
    result["provider_path"] = list(provider_path)
    return result


# ──────────────────────────────────────────────────────────────
# Strategy 1: Gemini AI Email Finder
# ──────────────────────────────────────────────────────────────

async def _search_gemini_email(company_name: str, domain: str) -> tuple[Optional[dict], str]:
    """Ask Gemini to provide a real HR/careers/contact email for the company."""
    if not gemini_enabled():
        return None, "gemini_not_enabled"

    try:
        prompt = (
            f"I need the HR or careers or general contact email address for the company \"{company_name}\" "
            f"(website: {domain}).\n\n"
            "Rules:\n"
            "1. Return ONLY one email address, nothing else. No explanation.\n"
            "2. Prefer HR/careers/talent/recruiting emails.\n"
            "3. If you don't know the exact email, return the most likely "
            f"general contact email for {domain} (like info@{domain} or contact@{domain}).\n"
            "4. The email domain MUST match or be related to the company.\n"
            "5. Return ONLY the email, e.g.: hr@company.com\n"
        )

        text = await gemini_generate_text(
            prompt,
            temperature=0,
            max_output_tokens=60,
        )

        text = text.strip().lower()
        # Extract email from response
        match = EMAIL_RE.search(text)
        if match:
            email = match.group(0)
            # Validate it's related to the company domain
            email_domain = email.split("@")[1] if "@" in email else ""
            # Accept if email domain matches or is a reasonable company domain
            if email_domain and "." in email_domain:
                confidence = 0.6
                if _contains_hr_keywords(email):
                    confidence = 0.75
                if domain in email_domain or email_domain in domain:
                    confidence += 0.1
                return {
                    "email": email,
                    "confidence": min(0.85, confidence),
                    "source": "gemini_ai",
                }, "gemini_email_found"

        return None, "gemini_no_valid_email"
    except Exception as exc:
        print(f"Gemini email finder error: {exc}")
        return None, "gemini_exception"


# ──────────────────────────────────────────────────────────────
# Strategy 2: Direct HTTP Website Scraping
# ──────────────────────────────────────────────────────────────

async def _scrape_website_emails(domain: str, company_name: str = "") -> tuple[Optional[dict], str]:
    """Fetch common pages on the company website and extract email addresses."""
    pages_to_try = [
        f"https://{domain}",
        f"https://{domain}/contact",
        f"https://{domain}/contact-us",
        f"https://{domain}/contactus",
        f"https://{domain}/about",
        f"https://{domain}/about-us",
        f"https://{domain}/careers",
        f"https://{domain}/jobs",
        f"https://{domain}/team",
        f"https://{domain}/company",
        f"https://{domain}/hiring",
        f"https://www.{domain}",
        f"https://www.{domain}/contact",
        f"https://www.{domain}/about",
        f"https://www.{domain}/careers",
    ]

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }

    all_emails: list[dict] = []
    seen: set[str] = set()

    async with httpx.AsyncClient(timeout=8, headers=headers, follow_redirects=True) as client:
        for url in pages_to_try:
            try:
                resp = await client.get(url)
                if resp.status_code >= 400:
                    continue
                text = resp.text or ""

                # Also extract from mailto: links
                mailto_matches = re.findall(r'mailto:([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})', text)
                regular_matches = EMAIL_RE.findall(text)

                for email in mailto_matches + regular_matches:
                    email = email.strip().lower()
                    if email in seen:
                        continue
                    if any(skip in email for skip in _JUNK_EMAIL_MARKERS):
                        continue
                    parts = email.split("@")
                    if len(parts) != 2 or "." not in parts[1]:
                        continue
                    seen.add(email)
                    conf = 0.6
                    if _contains_hr_keywords(email):
                        conf = 0.85
                    if domain in email:
                        conf += 0.1
                    all_emails.append({"email": email, "confidence": min(0.95, conf)})
            except Exception:
                continue

    if all_emails:
        best = _pick_best_candidate(all_emails)
        if best:
            return {
                "email": best["email"],
                "confidence": best["confidence"],
                "source": "website_scrape",
            }, "website_scrape_ok"

    return None, "website_scrape_no_emails"


# ──────────────────────────────────────────────────────────────
# Strategy 3: Apify Contact Info Scraper
# ──────────────────────────────────────────────────────────────

async def _search_apify_email(domain: str) -> tuple[Optional[dict], str]:
    """Use Apify's Contact Info Scraper to find emails from a company website."""
    from config import settings as _cfg
    if not _cfg.apify_enabled or not _cfg.apify_api_token or ApifyClient is None:
        return None, "apify_email_not_configured"

    try:
        import asyncio as _asyncio

        def _run_sync() -> list[dict]:
            client = ApifyClient(_cfg.apify_api_token)
            urls = [
                f"https://{domain}",
                f"https://{domain}/contact",
                f"https://{domain}/about",
                f"https://{domain}/careers",
                f"https://www.{domain}",
            ]

            # Try a lightweight contact info scraper
            actors_to_try = [
                {
                    "id": "vdrmota/contact-info-scraper",
                    "input": {
                        "urls": [{"url": u} for u in urls[:3]],
                        "maxRequestsPerStartUrl": 3,
                        "maxDepth": 1,
                    },
                },
            ]

            for actor_config in actors_to_try:
                try:
                    run = client.actor(actor_config["id"]).call(
                        run_input=actor_config["input"],
                        timeout_secs=60,
                    )
                    items: list[dict] = []
                    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
                        items.append(item)
                    return items
                except Exception as exc:
                    msg = str(exc).lower()
                    if "must rent" in msg or "free trial" in msg or "paid" in msg:
                        continue  # try next actor
                    print(f"Apify email scraper {actor_config['id']} error: {exc}")
                    return []
            return []

        items = await _asyncio.to_thread(_run_sync)

        if not items:
            return None, "apify_email_no_results"

        # Extract emails from Apify results
        all_emails: list[dict] = []
        seen: set[str] = set()
        for item in items:
            # Contact info scraper typically returns emails in various fields
            for key in ("emails", "email", "contactEmails"):
                raw = item.get(key)
                if isinstance(raw, str):
                    raw = [raw]
                if isinstance(raw, list):
                    for email_val in raw:
                        if isinstance(email_val, str):
                            email_val = email_val.strip().lower()
                        elif isinstance(email_val, dict):
                            email_val = str(email_val.get("value", email_val.get("email", ""))).strip().lower()
                        else:
                            continue
                        if not email_val or not EMAIL_RE.fullmatch(email_val):
                            continue
                        if email_val in seen:
                            continue
                        if any(skip in email_val for skip in _JUNK_EMAIL_MARKERS):
                            continue
                        seen.add(email_val)
                        conf = 0.6
                        if _contains_hr_keywords(email_val):
                            conf = 0.85
                        if domain in email_val:
                            conf += 0.1
                        all_emails.append({"email": email_val, "confidence": min(0.95, conf)})

        if all_emails:
            best = _pick_best_candidate(all_emails)
            if best:
                return {
                    "email": best["email"],
                    "confidence": best["confidence"],
                    "source": "apify_scraper",
                }, "apify_email_found"

        return None, "apify_email_no_emails_extracted"
    except Exception as exc:
        print(f"Apify email scraper error: {exc}")
        return None, "apify_email_exception"


# ──────────────────────────────────────────────────────────────
# Strategy 5: Smart Pattern Fallback (verified domain)
# ──────────────────────────────────────────────────────────────

async def _smart_pattern_fallback(domain: str) -> tuple[Optional[dict], str]:
    """
    Use common HR email patterns, but ONLY if:
    1. The domain has a working website
    2. The email passes SMTP verification (mailbox actually exists)
    """
    # First check if the domain has a working website
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/123.0.0.0",
    }
    domain_is_alive = False
    try:
        async with httpx.AsyncClient(timeout=5, headers=headers, follow_redirects=True) as client:
            resp = await client.head(f"https://{domain}")
            if resp.status_code < 500:
                domain_is_alive = True
    except Exception:
        try:
            async with httpx.AsyncClient(timeout=5, headers=headers, follow_redirects=True) as client:
                resp = await client.head(f"http://{domain}")
                if resp.status_code < 500:
                    domain_is_alive = True
        except Exception:
            pass

    if not domain_is_alive:
        return None, "pattern_domain_not_reachable"

    # Domain is alive — try each pattern with SMTP verification
    patterns_ranked = [
        (f"hr@{domain}", 0.45),
        (f"careers@{domain}", 0.40),
        (f"info@{domain}", 0.40),
        (f"contact@{domain}", 0.38),
        (f"jobs@{domain}", 0.35),
        (f"talent@{domain}", 0.35),
        (f"hiring@{domain}", 0.35),
    ]

    # Try each pattern — pick the first one that passes SMTP verification
    for email, confidence in patterns_ranked:
        if verify_email_smtp(email):
            return {
                "email": email,
                "confidence": min(0.75, confidence + 0.2),  # Boost: SMTP verified
                "source": "pattern_smtp_verified",
                "smtp_verified": True,
                "all_patterns": [p[0] for p in patterns_ranked],
            }, "pattern_smtp_verified"

    return None, "pattern_all_smtp_rejected"


# ──────────────────────────────────────────────────────────────
# DNS MX Verification (prevents bounced emails)
# ──────────────────────────────────────────────────────────────

def verify_email_domain(domain: str) -> bool:
    """
    Check if an email domain has MX records (can receive email).
    Returns True if the domain has MX records, False otherwise.
    """
    if not domain or "." not in domain:
        return False

    # Try MX records first
    try:
        if dns is not None:
            answers = dns.resolver.resolve(domain, "MX")
            return len(answers) > 0
    except Exception:
        pass

    # Fallback: try A record (some domains use A record for mail)
    try:
        if dns is not None:
            answers = dns.resolver.resolve(domain, "A")
            return len(answers) > 0
    except Exception:
        pass

    return False


def _get_mx_host(domain: str) -> Optional[str]:
    """Get the primary MX host for a domain."""
    try:
        if dns is not None:
            answers = dns.resolver.resolve(domain, "MX")
            if answers:
                # Return lowest priority MX
                best = min(answers, key=lambda r: r.preference)
                return str(best.exchange).rstrip(".")
    except Exception:
        pass
    return None


def verify_email_smtp(email: str) -> bool:
    """
    Verify that an email address actually exists using SMTP RCPT TO.
    Connects to the mail server and checks if the mailbox is valid.
    Returns True if the email appears valid, False if rejected.
    """
    import smtplib
    import socket

    if not email or "@" not in email:
        return False

    domain = email.split("@")[1]
    mx_host = _get_mx_host(domain)
    if not mx_host:
        return False

    try:
        smtp = smtplib.SMTP(timeout=10)
        smtp.connect(mx_host, 25)
        smtp.helo("jobhunter.local")
        smtp.mail("verify@jobhunter.local")
        code, _ = smtp.rcpt(email)
        smtp.quit()
        # 250 = OK, 251 = forwarded — both mean mailbox exists
        # 550, 551, 552, 553 = mailbox doesn't exist
        return code in (250, 251)
    except smtplib.SMTPServerDisconnected:
        # Some servers disconnect — treat as inconclusive, allow the email
        return True
    except (smtplib.SMTPConnectError, socket.timeout, socket.gaierror, OSError):
        # Connection failed — can't verify, allow the email (benefit of doubt)
        return True
    except Exception as exc:
        print(f"SMTP verify error for {email}: {exc}")
        # On any other error, allow the email
        return True


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _normalize_domain(domain: str) -> str:
    cleaned = domain.strip().lower()
    cleaned = cleaned.replace("http://", "").replace("https://", "").replace("www.", "")
    cleaned = cleaned.split("/")[0]
    cleaned = cleaned.split("?")[0]
    cleaned = cleaned.strip().strip(".")
    cleaned = re.sub(r"[^a-z0-9.-]", "", cleaned)
    cleaned = re.sub(r"\.{2,}", ".", cleaned)
    if cleaned and "." not in cleaned:
        cleaned = f"{cleaned}.com"
    return cleaned


def _to_float_confidence(value: Any, default: float = 0.5) -> float:
    try:
        num = float(value)
        if num > 1:
            return max(0.0, min(1.0, num / 100.0))
        return max(0.0, min(1.0, num))
    except Exception:
        return default


def _contains_hr_keywords(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in HR_KEYWORDS)


def _detect_quota_or_error(status_code: int, payload: Any) -> Optional[str]:
    blob = str(payload).lower()
    if status_code in (402, 429):
        return "quota_or_rate_limit"
    if "not_enough_credits" in blob or ("insufficient" in blob and "credit" in blob):
        return "quota_exhausted"
    if "limit" in blob and "credit" in blob:
        return "quota_exhausted"
    if status_code in (401, 403):
        return "auth_error"
    return None


def _pick_best_candidate(candidates: list[dict]) -> Optional[dict]:
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item.get("confidence", 0), reverse=True)[0]


def _extract_email_candidates(payload: Any) -> list[dict]:
    candidates: list[dict] = []
    seen: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            raw_email: Optional[str] = None
            for key in ("email", "value", "address"):
                value = node.get(key)
                if isinstance(value, str) and EMAIL_RE.fullmatch(value.strip()):
                    raw_email = value.strip().lower()
                    break

            if raw_email and raw_email not in seen:
                seen.add(raw_email)
                title_bits = [
                    str(node.get("position", "")),
                    str(node.get("job_title", "")),
                    str(node.get("department", "")),
                    str(node.get("type", "")),
                    raw_email,
                ]
                score = _to_float_confidence(node.get("confidence") or node.get("score"), default=0.5)
                if _contains_hr_keywords(" ".join(title_bits)):
                    score = min(0.99, score + 0.2)
                candidates.append({"email": raw_email, "confidence": score})

            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return candidates


# ──────────────────────────────────────────────────────────────
# Snov.io
# ──────────────────────────────────────────────────────────────

async def _search_snov(domain: str) -> tuple[Optional[dict], str]:
    if not settings.snov_client_id or not settings.snov_client_secret:
        return None, "snov_not_configured"

    async with httpx.AsyncClient(timeout=20) as client:
        token = await _snov_access_token(client)
        if not token:
            return None, "snov_auth_failed"

        task_hash, reason = await _snov_start_domain_search(client, token, domain)
        if not task_hash:
            return None, reason

        candidates, reason = await _snov_poll_domain_search(client, token, task_hash)
        if not candidates:
            return None, reason

        best = _pick_best_candidate(candidates)
        if not best:
            return None, "snov_no_usable_email"

        return {
            "email": best["email"],
            "confidence": best["confidence"],
            "source": "snov_io",
        }, "snov_ok"


async def _snov_access_token(client: httpx.AsyncClient) -> Optional[str]:
    try:
        response = await client.post(
            "https://api.snov.io/v1/oauth/access_token",
            data={
                "grant_type": "client_credentials",
                "client_id": settings.snov_client_id,
                "client_secret": settings.snov_client_secret,
            },
        )
        response.raise_for_status()
        return response.json().get("access_token")
    except Exception as exc:
        print(f"Snov auth error: {exc}")
        return None


async def _snov_start_domain_search(
    client: httpx.AsyncClient,
    token: str,
    domain: str,
) -> tuple[Optional[str], str]:
    try:
        response = await client.get(
            "https://api.snov.io/v2/domain-search/domain-emails/start",
            params={"access_token": token, "domain": domain},
        )
        payload = response.json()
        failure = _detect_quota_or_error(response.status_code, payload)
        if failure:
            return None, f"snov_{failure}"

        if response.status_code >= 400:
            return None, "snov_start_failed"

        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        task_hash = data.get("task_hash") or data.get("hash") or payload.get("task_hash")
        if not task_hash:
            return None, "snov_missing_task_hash"
        return str(task_hash), "snov_started"
    except Exception as exc:
        print(f"Snov start error: {exc}")
        return None, "snov_start_exception"


async def _snov_poll_domain_search(
    client: httpx.AsyncClient,
    token: str,
    task_hash: str,
    max_polls: int = 8,
) -> tuple[list[dict], str]:
    for _ in range(max_polls):
        try:
            response = await client.get(
                "https://api.snov.io/v2/domain-search/domain-emails/result",
                params={"access_token": token, "task_hash": task_hash},
            )
            payload = response.json()

            failure = _detect_quota_or_error(response.status_code, payload)
            if failure:
                return [], f"snov_{failure}"

            data = payload.get("data", {}) if isinstance(payload, dict) else {}
            task_status = str(data.get("task_status") or payload.get("task_status") or "").lower()

            candidates = _extract_email_candidates(payload)
            if candidates:
                return candidates, "snov_completed"

            if task_status in {"in_progress", "queued", "created", "pending"}:
                await asyncio.sleep(1.2)
                continue

            if task_status in {"failed", "error"}:
                return [], "snov_task_failed"

            if task_status in {"completed", "done", "success"}:
                return [], "snov_no_results"

            await asyncio.sleep(1.0)
        except Exception as exc:
            print(f"Snov poll error: {exc}")
            return [], "snov_result_exception"

    return [], "snov_timeout"


# ──────────────────────────────────────────────────────────────
# Prospeo Domain Search (simple, direct)
# ──────────────────────────────────────────────────────────────

async def _search_prospeo_domain(domain: str) -> tuple[Optional[dict], str]:
    """Use Prospeo's domain-search endpoint to find emails directly by domain.
    This is simpler and more reliable than the person-search + enrich flow."""
    if not settings.prospeo_api_key:
        return None, "prospeo_domain_not_configured"

    headers = {
        "Authorization": f"Bearer {settings.prospeo_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=20, headers=headers) as client:
            response = await client.post(
                "https://api.prospeo.io/domain-search",
                json={"domain": domain},
            )
            payload = response.json()

            failure = _detect_quota_or_error(response.status_code, payload)
            if failure:
                return None, f"prospeo_domain_{failure}"

            if isinstance(payload, dict) and payload.get("error"):
                error_code = payload.get("error_code", "api_error")
                # 'domain_not_found' is expected for smaller companies
                if "not_found" in str(error_code).lower():
                    return None, "prospeo_domain_not_found"
                return None, f"prospeo_domain_{error_code}"

            # Extract emails from response
            email_list = payload.get("response", {}).get("email_list", []) if isinstance(payload, dict) else []
            if not email_list and isinstance(payload, dict):
                # Try alternative response shapes
                email_list = payload.get("emails", []) or payload.get("results", [])

            candidates: list[dict] = []
            for entry in email_list:
                if not isinstance(entry, dict):
                    continue
                email_val = str(entry.get("email", "")).strip().lower()
                if not email_val or not EMAIL_RE.fullmatch(email_val):
                    continue
                if any(skip in email_val for skip in _JUNK_EMAIL_MARKERS):
                    continue

                title = str(entry.get("title", "") or entry.get("position", "")).lower()
                confidence = 0.7
                if _contains_hr_keywords(email_val) or _contains_hr_keywords(title):
                    confidence = 0.9
                verification = str(entry.get("verification", {}).get("status", "") if isinstance(entry.get("verification"), dict) else entry.get("email_status", "")).lower()
                if "valid" in verification:
                    confidence = min(0.95, confidence + 0.1)
                elif "invalid" in verification:
                    confidence = 0.2

                candidates.append({"email": email_val, "confidence": confidence})

            if candidates:
                best = _pick_best_candidate(candidates)
                if best:
                    return {
                        "email": best["email"],
                        "confidence": best["confidence"],
                        "source": "prospeo_domain",
                    }, "prospeo_domain_ok"

            return None, "prospeo_domain_no_emails"
    except Exception as exc:
        print(f"Prospeo domain search error: {exc}")
        return None, "prospeo_domain_exception"


# ──────────────────────────────────────────────────────────────
# Prospeo Person Search + Enrich (fallback)
# ──────────────────────────────────────────────────────────────

async def _search_prospeo(domain: str) -> tuple[Optional[dict], str]:
    if not settings.prospeo_api_key:
        return None, "prospeo_not_configured"

    headers = {
        "Authorization": f"Bearer {settings.prospeo_api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=20, headers=headers) as client:
        people, reason = await _prospeo_search_people(client, domain)
        if not people:
            if reason:
                print(f"Prospeo search skipped: {reason}")
            return None, reason

        ranked_people = sorted(
            people,
            key=lambda p: 1 if _contains_hr_keywords(f"{p.get('title', '')} {p.get('name', '')}") else 0,
            reverse=True,
        )

        last_reason = "prospeo_no_usable_email"
        for person in ranked_people[:8]:
            result, enrich_reason = await _prospeo_enrich_person(client, person.get("name", ""), domain)
            if result:
                return result, "prospeo_ok"
            last_reason = enrich_reason

        return None, last_reason


async def _prospeo_search_people(client: httpx.AsyncClient, domain: str) -> tuple[list[dict], str]:
    try:
        response = await client.post(
            "https://api.prospeo.io/search-person",
            json={
                "page": 1,
                "per_page": 25,
                "filters": {
                    "company": {
                        "websites": {
                            "include": [domain],
                        }
                    }
                },
            },
        )
        payload = response.json()

        failure = _detect_quota_or_error(response.status_code, payload)
        if failure:
            return [], f"prospeo_{failure}"

        if isinstance(payload, dict) and payload.get("error"):
            return [], f"prospeo_{payload.get('error_code', 'api_error')}"

        rows = payload.get("results", []) if isinstance(payload, dict) else []
        people: list[dict] = []
        for row in rows:
            person = row.get("person", {}) if isinstance(row, dict) else {}
            full_name = str(person.get("full_name") or "").strip()
            if not full_name:
                continue
            title = str(person.get("current_job_title") or person.get("headline") or "")
            people.append({"name": full_name, "title": title})

        if not people:
            return [], "prospeo_no_people"
        return people, "prospeo_people_found"
    except Exception as exc:
        print(f"Prospeo search error: {exc}")
        return [], "prospeo_search_exception"


async def _prospeo_enrich_person(
    client: httpx.AsyncClient,
    full_name: str,
    company_website: str,
) -> tuple[Optional[dict], str]:
    if not full_name:
        return None, "prospeo_no_name"

    try:
        response = await client.post(
            "https://api.prospeo.io/enrich-person",
            json={
                "data": {
                    "full_name": full_name,
                    "company_website": company_website,
                },
                "only_verified_email": False,
                "enrich_mobile": False,
            },
        )
        payload = response.json()

        failure = _detect_quota_or_error(response.status_code, payload)
        if failure:
            print(f"Prospeo enrich blocked: {failure}")
            return None, f"prospeo_{failure}"

        if isinstance(payload, dict) and payload.get("error"):
            return None, f"prospeo_{payload.get('error_code', 'api_error')}"

        email_value = _extract_prospeo_email(payload)
        if not email_value:
            return None, "prospeo_no_email"

        status_text = str(
            payload.get("person", {}).get("email", {}).get("status", "")
            if isinstance(payload, dict)
            else ""
        ).lower()
        confidence = 0.6
        if "valid" in status_text:
            confidence = 0.9
        elif "catch" in status_text:
            confidence = 0.75
        elif "invalid" in status_text:
            confidence = 0.2

        if _contains_hr_keywords(full_name):
            confidence = min(0.99, confidence + 0.05)

        return {
            "email": email_value,
            "confidence": confidence,
            "source": "prospeo",
        }, "prospeo_ok"
    except Exception as exc:
        print(f"Prospeo enrich error: {exc}")
        return None, "prospeo_enrich_exception"


def _extract_prospeo_email(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None

    person = payload.get("person", {})
    if isinstance(person, dict):
        email_block = person.get("email", {})
        if isinstance(email_block, dict):
            direct = email_block.get("email")
            if isinstance(direct, str) and EMAIL_RE.fullmatch(direct.strip()):
                return direct.strip().lower()
        direct = person.get("email")
        if isinstance(direct, str) and EMAIL_RE.fullmatch(direct.strip()):
            return direct.strip().lower()

    for key in ("email", "work_email", "business_email"):
        direct = payload.get(key)
        if isinstance(direct, str) and EMAIL_RE.fullmatch(direct.strip()):
            return direct.strip().lower()

    return None
