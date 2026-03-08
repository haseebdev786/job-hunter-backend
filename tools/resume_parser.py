from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import fitz
from docx import Document

from tools.gemini_tool import gemini_enabled, gemini_generate_json


def _extract_pdf_text(file_path: str) -> str:
    text_parts: list[str] = []
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text_parts.append(page.get_text())
    return "\n".join(text_parts).strip()


def _extract_docx_text(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs if p.text).strip()


def _safe_json_loads(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def _heuristic_extract_resume(text: str) -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    full_name = ""
    for line in lines[:8]:
        if "@" in line:
            continue
        if len(line.split()) >= 2 and len(line) <= 60 and re.fullmatch(r"[A-Za-z .'-]+", line):
            full_name = line
            break

    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phone_match = re.search(r"(\+?\d[\d\-\s()]{7,}\d)", text)

    summary = ""
    for idx, line in enumerate(lines):
        lower = line.lower()
        if lower.startswith("summary"):
            summary = line.split(":", 1)[-1].strip() if ":" in line else " ".join(lines[idx + 1 : idx + 3]).strip()
            break
    if not summary:
        summary = " ".join(lines[1:4]).strip()[:300] if len(lines) > 1 else ""

    skills: list[str] = []
    for line in lines:
        if "skill" in line.lower():
            raw = line.split(":", 1)[-1] if ":" in line else line
            pieces = [item.strip() for item in re.split(r"[,|/]", raw) if item.strip()]
            skills.extend(pieces)
            break
    seen: set[str] = set()
    unique_skills: list[str] = []
    for skill in skills:
        key = skill.lower()
        if key and key not in seen:
            seen.add(key)
            unique_skills.append(skill)

    return {
        "full_name": full_name,
        "email": email_match.group(0) if email_match else "",
        "phone": phone_match.group(0) if phone_match else "",
        "location": "",
        "summary": summary,
        "skills": unique_skills[:15],
        "experience": [],
        "education": [],
        "raw_text": text,
    }


async def _llm_extract_resume(text: str) -> dict:
    prompt = (
        "Extract structured information from this resume. Return ONLY valid JSON with no markdown backticks:\n"
        "{\n"
        '  "full_name": "...",\n'
        '  "email": "...",\n'
        '  "phone": "...",\n'
        '  "location": "...",\n'
        '  "portfolio_url": "...",\n'
        '  "github_url": "...",\n'
        '  "linkedin_url": "...",\n'
        '  "summary": "...",\n'
        '  "skills": ["skill1", "skill2"],\n'
        '  "experience": [\n'
        '    {"title": "...", "company": "...", "duration": "...", "bullets": ["..."]}\n'
        "  ],\n"
        '  "education": [\n'
        '    {"degree": "...", "institution": "...", "year": "..."}\n'
        "  ],\n"
        '  "raw_text": "full original resume text here"\n'
        "}\n\n"
        "CRITICAL RULES:\n"
        "1. You MUST ONLY extract Software Engineering, IT, Web Development, and Tech-related skills, experience, and projects.\n"
        "2. COMPLETELY IGNORE unrelated background like 'Electrician', 'Generator operator', etc. DO NOT include them in the summary, skills, or experience.\n"
        "3. Look carefully for developer skills (e.g. React.js, Next.js, PHP, Python, MongoDB) and Projects (e.g. AI Code Reviewer) and put them in 'skills' and 'experience' (treat projects as experience if needed).\n"
        "4. Extract any portfolio, github, or linkedin URLs found in the text.\n\n"
        f"RESUME TEXT:\n{text[:12000]}"
    )

    if not gemini_enabled():
        return _heuristic_extract_resume(text)

    try:
        parsed = await gemini_generate_json(
            prompt,
            default={},
            temperature=0.0,
            max_output_tokens=1800,
        )
        if not isinstance(parsed, dict) or not parsed:
            return _heuristic_extract_resume(text)
        parsed.setdefault("raw_text", text)
        return parsed
    except Exception as exc:
        print(f"Resume LLM extraction unavailable, using heuristic parser ({type(exc).__name__}).")
        return _heuristic_extract_resume(text)


async def parse_resume(file_path: str) -> dict:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        extracted_text = await asyncio.to_thread(_extract_pdf_text, file_path)
    elif suffix == ".docx":
        extracted_text = await asyncio.to_thread(_extract_docx_text, file_path)
    else:
        raise ValueError("Unsupported file format. Please upload PDF or DOCX.")

    if not extracted_text.strip():
        raise ValueError("Could not extract text from resume file.")

    return await _llm_extract_resume(extracted_text)
