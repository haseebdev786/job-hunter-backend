from __future__ import annotations

import json
import os
import re
import textwrap
from pathlib import Path
from typing import Any, Callable

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

from tools.gemini_tool import gemini_enabled, gemini_generate_json


def _safe_json(raw: str, default: dict) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").replace("json", "", 1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return default
        return default


async def _call_llm_json(system_prompt: str, user_prompt: str, default: dict) -> dict:
    if not gemini_enabled():
        return default

    try:
        parsed = await gemini_generate_json(
            user_prompt,
            default=default,
            system_prompt=system_prompt,
            temperature=0.0,
            max_output_tokens=1800,
        )
        if not isinstance(parsed, dict):
            return default
        return parsed
    except Exception:
        return default


def _safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)


def _write_wrapped(c: canvas.Canvas, text: str, x: int, y: int, width_chars: int = 100, line_height: int = 14) -> int:
    for line in textwrap.wrap(text, width=width_chars) or [""]:
        c.drawString(x, y, line)
        y -= line_height
    return y


def _render_tailored_resume_pdf(profile: dict, tailored: dict, pdf_path: str) -> None:
    c = canvas.Canvas(pdf_path, pagesize=LETTER)
    width, height = LETTER
    x = 50
    y = height - 50

    name = profile.get("full_name", "Candidate Name")
    email = profile.get("email", "")
    phone = profile.get("phone", "")
    location = profile.get("location", "")

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, name)
    y -= 22

    c.setFont("Helvetica", 10)
    contact = " | ".join([v for v in [email, phone, location] if v])
    y = _write_wrapped(c, contact, x, y, width_chars=95, line_height=12)
    y -= 10

    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Summary")
    y -= 16
    c.setFont("Helvetica", 10)
    y = _write_wrapped(c, tailored.get("tailored_summary", ""), x, y, width_chars=100, line_height=12)
    y -= 8

    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Skills")
    y -= 16
    c.setFont("Helvetica", 10)
    skills_text = ", ".join(tailored.get("skills_reordered", []))
    y = _write_wrapped(c, skills_text, x, y, width_chars=100, line_height=12)
    y -= 8

    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Experience")
    y -= 16
    c.setFont("Helvetica", 10)
    exp_bullets = tailored.get("experience_bullets", {}) or {}
    for company, bullets in exp_bullets.items():
        y = _write_wrapped(c, f"{company}", x, y, width_chars=100, line_height=12)
        for bullet in bullets[:4]:
            y = _write_wrapped(c, f"- {bullet}", x + 10, y, width_chars=95, line_height=12)
        y -= 4
        if y < 100:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica", 10)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Education")
    y -= 16
    c.setFont("Helvetica", 10)
    for edu in profile.get("education", [])[:3]:
        degree = edu.get("degree", "")
        institution = edu.get("institution", "")
        year = edu.get("year", "")
        y = _write_wrapped(c, f"{degree} - {institution} ({year})", x, y, width_chars=100, line_height=12)

    c.save()


async def run_resume_tailor_agent(state: dict[str, Any], emit: Callable) -> dict[str, Any]:
    """
    For each selected job:
    1. Tailor resume to job description
    2. Generate PDF with ReportLab
    3. Save file to uploads/resumes/{run_id}_{job_id}.pdf
    """
    run_id = state.get("run_id", "run")
    profile = state.get("user_profile", {})
    resume_raw_text = state.get("resume_raw_text", "")
    jobs = state.get("jobs", [])

    system_prompt = (
        "You are an expert resume writer. Tailor resumes to match job descriptions.\n"
        "STRICT RULES - NEVER BREAK:\n"
        "- NEVER fabricate experience, skills, companies, or education\n"
        "- NEVER change job titles, company names, or employment dates\n"
        "- ONLY rephrase and reorder existing content to better match the JD\n"
        "- Every fact must already exist in the original resume"
    )

    os.makedirs("uploads/resumes", exist_ok=True)
    tailored_resumes: dict[str, dict] = {}

    for job in jobs:
        job_id = str(job.get("job_id", ""))
        if not job_id:
            continue

        await emit(
            "resume_tailor",
            "ANALYZING_JD",
            {"job_title": job.get("title"), "company": job.get("company")},
            "running",
        )

        default = {
            "tailored_summary": profile.get("summary", ""),
            "skills_reordered": profile.get("skills", []),
            "experience_bullets": {exp.get("company", "Experience"): exp.get("bullets", []) for exp in profile.get("experience", [])[:3]},
            "keywords_matched": [],
            "match_score": 50,
        }

        user_prompt = (
            "Tailor this resume for the job below.\n\n"
            "JOB DESCRIPTION:\n"
            f"{job.get('description', '')[:3000]}\n\n"
            "ORIGINAL RESUME:\n"
            f"{resume_raw_text[:9000]}\n\n"
            "Return ONLY valid JSON, no markdown:\n"
            "{\n"
            '  "tailored_summary": "2-3 sentence professional summary",\n'
            '  "skills_reordered": ["most relevant skill", "..."],\n'
            '  "experience_bullets": {"Company Name": ["bullet 1", "bullet 2"]},\n'
            '  "keywords_matched": ["keyword1", "keyword2"],\n'
            '  "match_score": 85\n'
            "}"
        )

        await emit("resume_tailor", "TAILORING", {"job_id": job_id}, "running")
        tailored = await _call_llm_json(system_prompt, user_prompt, default)

        await emit(
            "resume_tailor",
            "TAILOR_PREVIEW",
            {
                "job_id": job_id,
                "company": job.get("company", ""),
                "job_title": job.get("title", ""),
                "tailored_summary": str(tailored.get("tailored_summary", "")),
                "skills_reordered": list(tailored.get("skills_reordered", []) or [])[:8],
                "keywords_matched": list(tailored.get("keywords_matched", []) or [])[:10],
                "match_score": float(tailored.get("match_score", 0.0) or 0.0),
            },
            "running",
        )

        await emit(
            "resume_tailor",
            "GENERATING_PDF",
            {"job_id": job_id, "company": job.get("company")},
            "running",
        )

        safe_job_id = _safe_filename(job_id or f"{job.get('company', 'job')}_{job.get('title', 'role')}")
        pdf_path = str(Path("uploads/resumes") / f"{run_id}_{safe_job_id}.pdf")

        try:
            _render_tailored_resume_pdf(profile, tailored, pdf_path)
        except Exception as exc:
            state.setdefault("errors", []).append(f"PDF generation failed for {job_id}: {exc}")
            pdf_path = ""

        tailored_resumes[job_id] = {
            "job_id": job_id,
            "content": tailored,
            "pdf_path": pdf_path,
            "match_score": float(tailored.get("match_score", 0.0)),
            "company": job.get("company", ""),
            "title": job.get("title", ""),
        }

        await emit(
            "resume_tailor",
            "RESUME_READY",
            {
                "job_id": job_id,
                "match_score": tailored_resumes[job_id]["match_score"],
                "pdf_path": pdf_path,
            },
            "done",
        )

    state["tailored_resumes"] = tailored_resumes
    return state
