from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from config import settings
from tools.gemini_tool import gemini_enabled, gemini_generate_json
from tools.gmail_tool import GmailTool
from tools.email_finder_tool import verify_email_domain


def _safe_json(raw: str, default: dict) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").replace("json", "", 1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return default
        return default


async def _draft_email_json(prompt: str, default: dict) -> dict:
    if not gemini_enabled():
        return default

    try:
        parsed = await gemini_generate_json(
            prompt,
            default=default,
            temperature=0.3,
            max_output_tokens=600,
        )
        if not isinstance(parsed, dict):
            return default
        return parsed
    except Exception:
        return default


def _load_gmail_credentials() -> dict[str, Any] | None:
    path = Path(settings.gmail_tokens_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_portfolio_links(profile: dict) -> dict[str, str]:
    """Extract portfolio, GitHub, LinkedIn links from profile."""
    links: dict[str, str] = {}
    raw_text = profile.get("raw_text", "")
    prefs = profile.get("_preferences", {})

    for key in ("portfolio_url", "portfolio", "website"):
        url = str(profile.get(key) or prefs.get(key) or "").strip()
        if url and url.startswith("http"):
            links["portfolio"] = url
            break

    for key in ("github_url", "github"):
        url = str(profile.get(key) or prefs.get(key) or "").strip()
        if url and url.startswith("http"):
            links["github"] = url
            break

    for key in ("linkedin_url", "linkedin"):
        url = str(profile.get(key) or prefs.get(key) or "").strip()
        if url and url.startswith("http"):
            links["linkedin"] = url
            break

    import re as _re
    for url_match in _re.finditer(r"https?://[^\s]+", raw_text):
        url = url_match.group(0).rstrip(".,;)")
        lower_url = url.lower()
        if "github.com" in lower_url and "github" not in links:
            links["github"] = url
        elif "linkedin.com" in lower_url and "linkedin" not in links:
            links["linkedin"] = url
        elif any(kw in lower_url for kw in ["portfolio", "vercel", "netlify", "github.io"]) and "portfolio" not in links:
            links["portfolio"] = url

    return links


def _build_links_footer(links: dict[str, str]) -> str:
    if not links:
        return ""
    parts = []
    if links.get("portfolio"):
        parts.append(f"Portfolio: {links['portfolio']}")
    if links.get("github"):
        parts.append(f"GitHub: {links['github']}")
    if links.get("linkedin"):
        parts.append(f"LinkedIn: {links['linkedin']}")
    return "\n".join(parts)


def _resolve_attachment(profile: dict, tailored: dict) -> str | None:
    """Return the original CV path for attachment. Fall back to tailored PDF."""
    original_cv = str(profile.get("_resume_path", "")).strip()
    if original_cv and Path(original_cv).exists():
        return original_cv
    pdf = tailored.get("pdf_path", "")
    if pdf and Path(pdf).exists():
        return pdf
    return None


def _build_draft_prompt(
    applicant_name: str,
    job_title: str,
    company: str,
    top_skills: list,
    tailored_summary: str,
    exp_text: str,
    keywords_matched: list,
    job_desc_snippet: str,
) -> str:
    return (
        "Write a professional job application email. Maximum 200 words. Be specific and compelling.\n\n"
        "STRICT RULES:\n"
        "- ONLY mention skills and experience that are relevant to THIS specific job\n"
        "- DO NOT mention any unrelated background (e.g., if applying for a dev role, don't mention electrician work)\n"
        "- Focus on the candidate's DEVELOPER/TECH skills and projects\n"
        "- DO NOT include phone numbers or email addresses in the body\n"
        "- DO NOT use placeholder text like '[Platform where you saw the job]'\n"
        "- Keep it concise, confident, and professional\n\n"
        f"APPLICANT: {applicant_name}\n"
        f"APPLYING FOR: {job_title} at {company}\n"
        f"RELEVANT SKILLS: {', '.join(top_skills[:6])}\n"
        f"TAILORED SUMMARY: {tailored_summary}\n"
        f"KEY EXPERIENCE: {exp_text[:500]}\n"
        f"MATCHED KEYWORDS: {', '.join(keywords_matched[:8])}\n"
        f"JOB DESCRIPTION SNIPPET: {job_desc_snippet[:400]}\n\n"
        "Return ONLY valid JSON, no markdown:\n"
        "{\n"
        f'  "subject": "Application: {job_title} - {applicant_name}",\n'
        '  "body": "full email body here (do NOT include links footer, it will be added automatically)"\n'
        "}\n\n"
        "STRUCTURE: Direct opening stating the role → 2-3 specific fit points with skills/projects → brief CTA → professional close with name"
    )


# ───────────────────────────────────────────────────────────
# Phase 2: Email DRAFTER — draft only, no sending
# ───────────────────────────────────────────────────────────
async def run_email_drafter_agent(state: dict[str, Any], emit: Callable) -> dict[str, Any]:
    """
    Draft personalized emails for each job with a tailored resume.
    Does NOT send — only creates drafts for user review.
    """
    profile = state.get("user_profile", {})
    jobs = state.get("jobs", [])
    hr_map = {entry.get("job_id"): entry for entry in state.get("hr_emails", [])}
    tailored_map = state.get("tailored_resumes", {})

    links = _extract_portfolio_links(profile)
    links_footer = _build_links_footer(links)
    applicant_name = profile.get("full_name", "Candidate")

    draft_emails: list[dict] = []

    for job in jobs:
        job_id = str(job.get("job_id", ""))
        hr = hr_map.get(job_id)
        tailored = tailored_map.get(job_id)
        if not hr or not tailored:
            continue

        recipient = hr.get("email", "")
        company = job.get("company", "Hiring Team")
        job_title = job.get("title", "the role")
        job_desc_snippet = job.get("description", "")[:600]

        # Skip low confidence emails
        email_confidence = float(hr.get("confidence", 0.0))
        if email_confidence < 0.35:
            await emit(
                "email_sender",
                "EMAIL_SKIPPED",
                {
                    "job_id": job_id,
                    "company": company,
                    "recipient": recipient,
                    "reason": f"Email confidence too low ({email_confidence:.0%})",
                },
                "running",
            )
            continue

        # Verify domain
        if recipient and "@" in recipient:
            recipient_domain = recipient.split("@")[1]
            if not verify_email_domain(recipient_domain):
                await emit(
                    "email_sender",
                    "EMAIL_SKIPPED",
                    {
                        "job_id": job_id,
                        "company": company,
                        "recipient": recipient,
                        "reason": f"Domain {recipient_domain} has no MX records",
                    },
                    "running",
                )
                continue

        await emit(
            "email_sender",
            "DRAFTING_EMAIL",
            {"company": company, "recipient": recipient},
            "running",
        )

        tailored_content = tailored.get("content", {})
        tailored_summary = tailored_content.get("tailored_summary", "")
        top_skills = (tailored_content.get("skills_reordered", []) or profile.get("skills", []))[:6]
        experience_bullets = tailored_content.get("experience_bullets", {})
        keywords_matched = tailored_content.get("keywords_matched", [])

        exp_text = ""
        for comp, bullets in (experience_bullets or {}).items():
            if bullets:
                exp_text += f"{comp}: {'; '.join(bullets[:2])}. "

        skills_str = ", ".join(top_skills[:4]) if top_skills else "software development"
        default_email = {
            "subject": f"Application: {job_title} - {applicant_name}",
            "body": (
                f"Dear Hiring Team at {company},\n\n"
                f"I am writing to express my interest in the {job_title} position. "
                f"With experience in {skills_str}, I am confident in my ability to contribute to your team.\n\n"
                f"{tailored_summary}\n\n"
                f"I have attached my resume for your review and would welcome the opportunity to discuss this role further."
                f"{chr(10) + chr(10) + links_footer if links_footer else ''}"
                f"\n\nBest regards,\n{applicant_name}"
            ),
        }

        prompt = _build_draft_prompt(
            applicant_name, job_title, company, top_skills,
            tailored_summary, exp_text, keywords_matched, job_desc_snippet,
        )

        draft = await _draft_email_json(prompt, default_email)
        subject = draft.get("subject", default_email["subject"])
        body = draft.get("body", default_email["body"])

        if links_footer and links_footer not in body:
            body = body.rstrip() + "\n\n" + links_footer

        attachment_path = _resolve_attachment(profile, tailored)

        await emit(
            "email_sender",
            "EMAIL_PREVIEW",
            {
                "job_id": job_id,
                "to": recipient,
                "subject": subject,
                "body": body,
                "attachment": attachment_path or "",
                "company": company,
                "title": job_title,
            },
            "running",
        )

        draft_emails.append(
            {
                "job_id": job_id,
                "recipient": recipient,
                "subject": subject,
                "body": body,
                "status": "draft",
                "attachment_path": attachment_path or "",
                "company": company,
                "title": job_title,
            }
        )

    state["draft_emails"] = draft_emails
    return state


# ───────────────────────────────────────────────────────────
# Phase 3: Email SENDER — sends only approved emails
# ───────────────────────────────────────────────────────────
async def run_email_sender_agent(state: dict[str, Any], emit: Callable) -> dict[str, Any]:
    """
    Send only approved emails. Uses edited content from EmailApproval if user made changes.
    The 'approved_emails' key in state should be populated by the API from the email_approvals table.
    """
    approved_emails = state.get("approved_emails", [])
    dry_run = bool(state.get("dry_run", True))

    gmail = GmailTool()
    credentials = _load_gmail_credentials() if not dry_run else None
    sent_emails: list[dict] = []

    for email_data in approved_emails:
        job_id = email_data.get("job_id", "")
        recipient = email_data.get("recipient", "")
        subject = email_data.get("subject", "")
        body = email_data.get("body", "")
        company = email_data.get("company", "")
        job_title = email_data.get("job_title", "")
        attachment_path = email_data.get("attachment_path", "")

        await emit(
            "email_sender",
            "SENDING",
            {"job_id": job_id, "to": recipient, "company": company},
            "running",
        )

        status = "preview"
        if not dry_run:
            sent = False
            try:
                if not credentials:
                    state.setdefault("errors", []).append(
                        "Gmail credentials not found. Reconnect Gmail from setup."
                    )
                else:
                    sent = await gmail.send_email(
                        credentials,
                        recipient,
                        subject,
                        body,
                        attachment_path if attachment_path and Path(attachment_path).exists() else None,
                    )
            except Exception as exc:
                state.setdefault("errors", []).append(f"Gmail send failed for {job_id}: {exc}")

            if sent:
                status = "sent"
                await emit(
                    "email_sender",
                    "EMAIL_SENT",
                    {"job_id": job_id, "company": company, "recipient": recipient},
                    "done",
                )
            else:
                status = "failed"
                await emit(
                    "email_sender",
                    "EMAIL_FAILED",
                    {"job_id": job_id, "company": company, "recipient": recipient},
                    "error",
                )
        else:
            status = "preview"
            await emit(
                "email_sender",
                "EMAIL_SENT",
                {"job_id": job_id, "company": company, "recipient": recipient, "dry_run": True},
                "done",
            )

        sent_emails.append(
            {
                "job_id": job_id,
                "recipient": recipient,
                "subject": subject,
                "body": body,
                "status": status,
                "attachment_path": attachment_path,
                "company": company,
                "title": job_title,
            }
        )

    state["sent_emails"] = sent_emails
    return state
