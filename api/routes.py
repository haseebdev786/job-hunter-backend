from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.orchestrator import (
    AgentState,
    run_orchestrator,
    run_orchestrator_phase1_with_jobs,
    run_orchestrator_phase2,
    run_orchestrator_phase3,
)
from config import settings
from models.database import (
    CVApproval,
    EmailApproval,
    HREmail,
    Job,
    Profile,
    Run,
    SentEmail,
    Step,
    TailoredResume,
    async_session,
    get_db,
)
from models.schemas import RunCreateRequest, RunCreateResponse
from tools.gmail_tool import GmailTool
from tools.gemini_tool import gemini_enabled, gemini_generate_json
from tools.resume_parser import parse_resume
from api.sse import stream_run_events

router = APIRouter()
RUN_TASKS: dict[str, asyncio.Task] = {}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[dict[str, str]] = Field(default_factory=list)


class CVEditRequest(BaseModel):
    content: dict = Field(default_factory=dict)


class EmailEditRequest(BaseModel):
    subject: str = ""
    body: str = ""


class CVChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    current_content: dict = Field(default_factory=dict)
    job_description: str = ""


class EmailChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    current_subject: str = ""
    current_body: str = ""
    job_title: str = ""
    company: str = ""
    raw_job_id: str = ""


class EnhanceCVRequest(BaseModel):
    jd_text: str = Field(default="", max_length=10000)
    jd_link: str = ""


# ─── Existing helpers ───

def _fallback_chat_decision(message: str) -> dict[str, str]:
    text = message.strip()
    if not text:
        return {"reply": "Your message is empty. Please type it again.", "action": "none"}
    lower = text.lower()

    if re.search(r"^(hi|hello|hey|salam|aoa|assalamualaikum)\b", lower):
        return {"reply": "Hi, welcome. I can help you with job search and applications.", "action": "none"}

    if re.search(r"\b(kya|what)\b.*\b(kar sakta|kar sakte|can you do)\b", lower) or re.search(
        r"\btum\b.*\bkar\b.*\bho\b",
        lower,
    ):
        return {
            "reply": "I can chat normally and also start a guided job-search/application run when you ask.",
            "action": "none",
        }

    start_intent = bool(
        re.search(
            r"\b(start|run|job search|job find|find jobs|apply|application|hunt)\b",
            lower,
        )
    )

    if start_intent:
        return {
            "reply": "Great, let's begin. I will collect details step by step.",
            "action": "start_job_intake",
        }

    return {"reply": "Understood. We are in chat mode. You can continue with your question.", "action": "none"}


async def _llm_chat_decision(message: str, history: list[dict[str, str]]) -> dict[str, str]:
    if not gemini_enabled():
        return _fallback_chat_decision(message)

    chat_history: list[dict[str, str]] = []
    for item in history[-8:]:
        role = "assistant" if str(item.get("role", "")).lower() == "assistant" else "user"
        text = str(item.get("text", "")).strip()
        if text:
            chat_history.append({"role": role, "content": text})

    prompt = (
        "Analyze the user message and return a decision in JSON.\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "reply": "short helpful reply in English",\n'
        '  "action": "none OR start_job_intake OR start_cv_intake"\n'
        "}\n\n"
        "Rules:\n"
        "- If the user wants to upload a new CV/resume or asks for CV-based job matching, set action=start_cv_intake.\n"
        "- If the user clearly wants to start job search/apply flow without CV upload context, set action=start_job_intake.\n"
        "- Otherwise set action=none.\n"
        "- Keep reply concise (1-3 lines).\n"
        "- If action starts a flow, mention that you'll collect details step by step.\n\n"
        f"USER_MESSAGE:\n{message}"
    )

    default = _fallback_chat_decision(message)
    try:
        parsed = await gemini_generate_json(
            prompt,
            default=default,
            history=chat_history,
            temperature=0.2,
            max_output_tokens=320,
            system_prompt=(
                "You are the conversation brain for a job-hunter app. "
                "Decide whether to start guided job-intake flow."
            ),
        )
        if not isinstance(parsed, dict):
            return default
        reply = str(parsed.get("reply", "")).strip() or default["reply"]
        action = str(parsed.get("action", "none")).strip().lower()
        if action not in {"none", "start_job_intake", "start_cv_intake"}:
            action = "none"
        return {"reply": reply, "action": action}
    except Exception:
        return default


def _raw_job_id(db_job_id: str) -> str:
    parts = db_job_id.split(":")
    if len(parts) >= 3:
        return parts[1]
    return db_job_id


async def _save_profile(db: AsyncSession, profile_data: dict[str, Any]) -> None:
    profile = await db.get(Profile, 1)
    if profile is None:
        profile = Profile(id=1, data=profile_data, updated_at=datetime.utcnow())
        db.add(profile)
    else:
        profile.data = profile_data
        profile.updated_at = datetime.utcnow()
    await db.commit()


async def _get_profile(db: AsyncSession) -> dict[str, Any] | None:
    profile = await db.get(Profile, 1)
    if not profile:
        return None
    return profile.data


async def _emit_step(run_id: str, agent: str, event_type: str, data: dict[str, Any], status: str = "running") -> None:
    logging.debug(f"Emitting step: run_id={run_id}, agent={agent}, event_type={event_type}, status={status}, data={data}")
    async with async_session() as db:
        try:
            step = Step(
                run_id=run_id,
                timestamp=datetime.utcnow(),
                agent=agent,
                event_type=event_type,
                data=data,
                status=status,
            )
            db.add(step)
            await db.commit()
            logging.debug(f"Step emitted successfully: {step.to_dict()}")
        except Exception as e:
            logging.error(f"Failed to emit step: {e}")
            raise


async def _persist_run_outputs(run_id: str, result_state: dict[str, Any]) -> None:
    logging.debug(f"Persisting run outputs for run_id={run_id}")
    async with async_session() as db:
        try:
            run = await db.get(Run, run_id)
            if run is None:
                logging.error(f"Run {run_id} not found in DB during persist operation.")
                return

            await db.execute(delete(Job).where(Job.run_id == run_id))
            await db.execute(delete(HREmail).where(HREmail.run_id == run_id))
            await db.execute(delete(TailoredResume).where(TailoredResume.run_id == run_id))
            await db.execute(delete(CVApproval).where(CVApproval.run_id == run_id))

            job_id_map: dict[str, str] = {}
            jobs_by_raw_id: dict[str, dict[str, Any]] = {}
            for idx, job in enumerate(result_state.get("jobs", [])):
                raw_job_id = str(job.get("job_id") or idx)
                db_job_id = f"{run_id}:{raw_job_id}:{idx}"
                job_id_map[raw_job_id] = db_job_id
                jobs_by_raw_id[raw_job_id] = job
                db.add(
                    Job(
                        id=db_job_id,
                        run_id=run_id,
                        title=job.get("title", ""),
                        company=job.get("company", ""),
                        location=job.get("location", ""),
                        description=job.get("description", ""),
                        apply_url=job.get("apply_url", ""),
                        source=job.get("source", ""),
                        relevance_score=float(job.get("relevance_score", 0.0)),
                    )
                )

            for item in result_state.get("hr_emails", []):
                raw_job_id = str(item.get("job_id", ""))
                mapped_job_id = job_id_map.get(raw_job_id)
                if not mapped_job_id:
                    continue
                db.add(
                    HREmail(
                        run_id=run_id,
                        job_id=mapped_job_id,
                        company=item.get("company", ""),
                        email=item.get("email", ""),
                        confidence=float(item.get("confidence", 0.0)),
                        source=item.get("source", ""),
                    )
                )

            tailored_resumes = result_state.get("tailored_resumes", {}) or {}
            tailored_entries: list[tuple[str, dict[str, Any]]] = []
            if isinstance(tailored_resumes, dict):
                tailored_entries = [
                    (str(raw_id), value if isinstance(value, dict) else {})
                    for raw_id, value in tailored_resumes.items()
                ]
            elif isinstance(tailored_resumes, list):
                for idx, value in enumerate(tailored_resumes):
                    if not isinstance(value, dict):
                        continue
                    raw_id = str(value.get("job_id") or idx)
                    tailored_entries.append((raw_id, value))

            persisted_cvs = 0
            skip_cv_approval = bool((run.summary or {}).get("skip_cv_approval", False))
            for raw_job_id, tailored in tailored_entries:
                mapped_job_id = job_id_map.get(raw_job_id)
                if not mapped_job_id:
                    continue

                content = tailored.get("content", {})
                if not isinstance(content, dict):
                    content = {}
                match_score = float(tailored.get("match_score", 0.0) or 0.0)
                pdf_path = str(tailored.get("pdf_path", "") or "")
                job_info = jobs_by_raw_id.get(raw_job_id, {})

                db.add(
                    TailoredResume(
                        run_id=run_id,
                        job_id=mapped_job_id,
                        content=json.dumps(content),
                        pdf_path=pdf_path,
                        match_score=match_score,
                    )
                )

                db.add(
                    CVApproval(
                        run_id=run_id,
                        job_id=raw_job_id,
                        status="approved" if skip_cv_approval else "pending",
                        original_content=content,
                        edited_content={},
                        company=job_info.get("company", tailored.get("company", "")),
                        job_title=job_info.get("title", tailored.get("title", "")),
                        match_score=match_score,
                        pdf_path=pdf_path,
                        updated_at=datetime.utcnow(),
                    )
                )
                persisted_cvs += 1

            if skip_cv_approval:
                run.phase = "phase2"
                run.status = "running"
            else:
                run.phase = "awaiting_cv_approval"
                run.status = "awaiting_cv_approval"
            summary = run.summary or {}
            summary["jobs_found"] = len(result_state.get("jobs", []))
            summary["emails_found"] = len(result_state.get("hr_emails", []))
            summary["cvs_tailored"] = persisted_cvs
            summary["resumes_tailored"] = persisted_cvs
            if skip_cv_approval:
                summary["cv_approval_skipped"] = True
                summary["cvs_auto_approved"] = persisted_cvs
            if "dry_run" not in summary:
                summary["dry_run"] = bool(result_state.get("dry_run", True))
            run.summary = summary

            await db.commit()
            logging.debug(f"Run outputs persisted successfully for run_id={run_id}")
        except Exception as e:
            logging.error(f"Failed to persist run outputs for run_id={run_id}: {e}")
            raise


async def _persist_phase2_outputs(run_id: str, result_state: dict[str, Any]) -> None:
    """Persist email drafts after Phase 2."""
    async with async_session() as db:
        run = await db.get(Run, run_id)
        if run is None:
            return

        summary = run.summary or {}
        skip_email_approval = bool(summary.get("skip_email_approval", False))
        drafts = result_state.get("draft_emails", [])

        # Create email approvals from draft emails
        await db.execute(delete(EmailApproval).where(EmailApproval.run_id == run_id))
        for draft in drafts:
            db.add(
                EmailApproval(
                    run_id=run_id,
                    job_id=str(draft.get("job_id", "")),
                    status="approved" if skip_email_approval else "pending",
                    recipient=draft.get("recipient", ""),
                    original_subject=draft.get("subject", ""),
                    original_body=draft.get("body", ""),
                    edited_subject="",
                    edited_body="",
                    company=draft.get("company", ""),
                    job_title=draft.get("title", ""),
                    attachment_path=draft.get("attachment_path", ""),
                    updated_at=datetime.utcnow(),
                )
            )

        drafted_count = len(drafts)
        summary["emails_drafted"] = drafted_count
        if drafted_count == 0:
            run.phase = "done"
            run.status = "done"
            summary["message"] = "No email drafts generated. Pipeline complete."
            summary["emails_processed"] = 0
            summary["emails_sent"] = 0
        elif skip_email_approval:
            run.phase = "phase3"
            run.status = "running"
            summary["email_approval_skipped"] = True
            summary["emails_auto_approved"] = drafted_count
        else:
            run.phase = "awaiting_email_approval"
            run.status = "awaiting_email_approval"
        run.summary = summary
        await db.commit()


async def _persist_phase3_outputs(run_id: str, result_state: dict[str, Any]) -> None:
    """Persist sent emails after Phase 3."""
    async with async_session() as db:
        run = await db.get(Run, run_id)
        if run is None:
            return

        await db.execute(delete(SentEmail).where(SentEmail.run_id == run_id))

        # Map raw job IDs to DB job IDs
        jobs_result = await db.execute(select(Job).where(Job.run_id == run_id))
        all_jobs = jobs_result.scalars().all()
        job_id_map = {}
        for j in all_jobs:
            raw = _raw_job_id(j.id)
            job_id_map[raw] = j.id

        for sent in result_state.get("sent_emails", []):
            raw_job_id = str(sent.get("job_id", ""))
            mapped_job_id = job_id_map.get(raw_job_id, raw_job_id)
            db.add(
                SentEmail(
                    run_id=run_id,
                    job_id=mapped_job_id,
                    recipient=sent.get("recipient", ""),
                    subject=sent.get("subject", ""),
                    body=sent.get("body", ""),
                    sent_at=datetime.utcnow(),
                    status=sent.get("status", "preview"),
                )
            )

        run.phase = "done"
        run.status = "done"
        summary = run.summary or {}
        summary["emails_processed"] = len(result_state.get("sent_emails", []))
        summary["emails_sent"] = sum(1 for e in result_state.get("sent_emails", []) if e.get("status") == "sent")
        run.summary = summary
        await db.commit()


# ─── Phase 1 execution (search → find → tailor → PAUSE) ───

async def _execute_run(run_id: str, dry_run: bool) -> None:
    async with async_session() as db:
        run = await db.get(Run, run_id)
        if not run:
            return
        run.status = "running"
        run.phase = "phase1"
        await db.commit()
        profile = run.user_profile if isinstance(run.user_profile, dict) else {}
        prefs = run.preferences if isinstance(run.preferences, dict) else {}

    initial_state: AgentState = {
        "run_id": run_id,
        "user_profile": profile,
        "preferences": prefs,
        "resume_raw_text": profile.get("raw_text", ""),
        "jobs": [],
        "hr_emails": [],
        "tailored_resumes": {},
        "sent_emails": [],
        "draft_emails": [],
        "steps": [],
        "current_agent": "orchestrator",
        "dry_run": dry_run,
        "phase": "phase1",
        "errors": [],
        "skip_cv_approval": bool((run.summary or {}).get("skip_cv_approval", False)),
        "skip_email_approval": bool((run.summary or {}).get("skip_email_approval", False)),
    }

    async def emit(agent: str, event_type: str, data: dict[str, Any], status: str = "running") -> None:
        await _emit_step(run_id, agent, event_type, data, status)

    try:
        result = await run_orchestrator(initial_state, emit)
        await _persist_run_outputs(run_id, result)
        if bool((result or {}).get("skip_cv_approval", initial_state.get("skip_cv_approval", False))):
            await _execute_phase2(run_id)
    except Exception as exc:
        await _emit_step(run_id, "orchestrator", "RUN_CRASHED", {"error": str(exc)}, "error")
        async with async_session() as db:
            run = await db.get(Run, run_id)
            if run:
                run.status = "failed"
                run.summary = {"error": str(exc)}
                await db.commit()


# ─── Phase 2 execution (draft emails → PAUSE) ───

async def _execute_phase2(run_id: str) -> None:
    async with async_session() as db:
        run = await db.get(Run, run_id)
        if not run:
            return
        run.status = "running"
        run.phase = "phase2"
        await db.commit()
        profile = run.user_profile if isinstance(run.user_profile, dict) else {}
        prefs = run.preferences if isinstance(run.preferences, dict) else {}

    # Rebuild state from DB
    async with async_session() as db:
        jobs_result = await db.execute(select(Job).where(Job.run_id == run_id))
        db_jobs = jobs_result.scalars().all()

        hr_result = await db.execute(select(HREmail).where(HREmail.run_id == run_id))
        db_hr_emails = hr_result.scalars().all()

        resume_result = await db.execute(select(TailoredResume).where(TailoredResume.run_id == run_id))
        db_resumes = resume_result.scalars().all()

        cv_result = await db.execute(
            select(CVApproval).where(CVApproval.run_id == run_id, CVApproval.status == "approved")
        )
        approved_cvs = cv_result.scalars().all()

    # Convert DB objects to state dicts
    approved_job_ids = {cv.job_id for cv in approved_cvs}

    jobs = [
        {
            "job_id": _raw_job_id(j.id),
            "title": j.title,
            "company": j.company,
            "location": j.location,
            "description": j.description,
            "apply_url": j.apply_url,
            "source": j.source,
        }
        for j in db_jobs
        if _raw_job_id(j.id) in approved_job_ids
    ]

    hr_emails = [
        {
            "job_id": _raw_job_id(e.job_id),
            "company": e.company,
            "email": e.email,
            "confidence": e.confidence,
            "source": e.source,
        }
        for e in db_hr_emails
        if _raw_job_id(e.job_id) in approved_job_ids
    ]

    tailored_resumes = {}
    for r in db_resumes:
        raw_id = _raw_job_id(r.job_id)
        if raw_id in approved_job_ids:
            # Use edited content from CV approval if available
            cv_approval = next((cv for cv in approved_cvs if cv.job_id == raw_id), None)
            content = cv_approval.edited_content if cv_approval and cv_approval.edited_content else json.loads(r.content) if r.content else {}
            tailored_resumes[raw_id] = {
                "content": content,
                "pdf_path": r.pdf_path,
                "match_score": r.match_score,
            }

    state: AgentState = {
        "run_id": run_id,
        "user_profile": profile,
        "preferences": prefs,
        "resume_raw_text": profile.get("raw_text", ""),
        "jobs": jobs,
        "hr_emails": hr_emails,
        "tailored_resumes": tailored_resumes,
        "sent_emails": [],
        "draft_emails": [],
        "steps": [],
        "current_agent": "orchestrator",
        "dry_run": bool(run.summary.get("dry_run", True) if run.summary else True),
        "phase": "phase2",
        "errors": [],
        "skip_cv_approval": bool(run.summary.get("skip_cv_approval", False) if run.summary else False),
        "skip_email_approval": bool(run.summary.get("skip_email_approval", False) if run.summary else False),
    }

    async def emit(agent: str, event_type: str, data: dict[str, Any], status: str = "running") -> None:
        await _emit_step(run_id, agent, event_type, data, status)

    try:
        result = await run_orchestrator_phase2(state, emit)
        await _persist_phase2_outputs(run_id, result)
        drafted_count = len(result.get("draft_emails", []) if isinstance(result, dict) else [])
        if drafted_count > 0 and bool(state.get("skip_email_approval", False)):
            await _execute_phase3(run_id)
    except Exception as exc:
        await _emit_step(run_id, "orchestrator", "RUN_CRASHED", {"error": str(exc)}, "error")
        async with async_session() as db:
            run = await db.get(Run, run_id)
            if run:
                run.status = "failed"
                await db.commit()


# ─── Phase 3 execution (send approved emails) ───

async def _execute_phase3(run_id: str) -> None:
    async with async_session() as db:
        run = await db.get(Run, run_id)
        if not run:
            return
        run.status = "running"
        run.phase = "phase3"
        await db.commit()

    async with async_session() as db:
        email_result = await db.execute(
            select(EmailApproval).where(EmailApproval.run_id == run_id, EmailApproval.status == "approved")
        )
        approved_emails = email_result.scalars().all()

        run = await db.get(Run, run_id)
        profile = run.user_profile if run and isinstance(run.user_profile, dict) else {}
        prefs = run.preferences if run and isinstance(run.preferences, dict) else {}
        dry_run = bool(run.summary.get("dry_run", True) if run and run.summary else True)

    # Build approved emails list for the sender
    approved_email_dicts = [
        {
            "job_id": e.job_id,
            "recipient": e.recipient,
            "subject": e.edited_subject or e.original_subject,
            "body": e.edited_body or e.original_body,
            "company": e.company,
            "job_title": e.job_title,
            "attachment_path": e.attachment_path,
        }
        for e in approved_emails
    ]

    state: AgentState = {
        "run_id": run_id,
        "user_profile": profile,
        "preferences": prefs,
        "resume_raw_text": profile.get("raw_text", ""),
        "jobs": [],
        "hr_emails": [],
        "tailored_resumes": {},
        "sent_emails": [],
        "draft_emails": [],
        "approved_emails": approved_email_dicts,
        "steps": [],
        "current_agent": "orchestrator",
        "dry_run": dry_run,
        "phase": "phase3",
        "errors": [],
        "skip_cv_approval": bool(run.summary.get("skip_cv_approval", False) if run and run.summary else False),
        "skip_email_approval": bool(run.summary.get("skip_email_approval", False) if run and run.summary else False),
    }

    async def emit(agent: str, event_type: str, data: dict[str, Any], status: str = "running") -> None:
        await _emit_step(run_id, agent, event_type, data, status)

    try:
        result = await run_orchestrator_phase3(state, emit)
        await _persist_phase3_outputs(run_id, result)
    except Exception as exc:
        await _emit_step(run_id, "orchestrator", "RUN_CRASHED", {"error": str(exc)}, "error")
        async with async_session() as db:
            run = await db.get(Run, run_id)
            if run:
                run.status = "failed"
                await db.commit()


# ═══════════════════════════════════════════════
# EXISTING ENDPOINTS
# ═══════════════════════════════════════════════

@router.post("/profile")
async def save_profile(
    resume: UploadFile = File(...),
    preferences: str | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    os.makedirs(settings.upload_dir, exist_ok=True)
    upload_path = Path(settings.upload_dir) / f"{uuid4()}_{resume.filename}"

    content = await resume.read()
    upload_path.write_bytes(content)

    try:
        parsed = await parse_resume(str(upload_path))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse resume: {exc}") from exc
    extra_prefs = json.loads(preferences) if preferences else {}
    parsed["_preferences"] = extra_prefs
    parsed["_resume_path"] = str(upload_path)

    await _save_profile(db, parsed)
    return {"profile": parsed}


@router.get("/profile")
async def get_profile(db: AsyncSession = Depends(get_db)):
    profile = await _get_profile(db)
    return {"profile": profile or {}}


@router.post("/chat")
async def chat(payload: ChatRequest):
    decision = await _llm_chat_decision(payload.message, payload.history)
    return decision


@router.post("/run", response_model=RunCreateResponse)
async def start_run(payload: RunCreateRequest, db: AsyncSession = Depends(get_db)):
    profile = await _get_profile(db)
    if not profile:
        raise HTTPException(status_code=400, detail="Profile not found. Upload resume first.")

    running_result = await db.execute(select(Run).where(Run.status == "running").limit(1))
    running = running_result.scalar_one_or_none()

    if running:
        task = RUN_TASKS.get(running.id)
        if not task or task.done():
            running.status = "failed"
            running.summary = {"error": "Run was interrupted by server restart or silent crash."}
            await db.commit()
        else:
            raise HTTPException(
                status_code=409,
                detail=f"Another run is already in progress ({running.id}). Wait for completion before starting a new run.",
            )

    run_id = str(uuid4())
    run = Run(
        id=run_id,
        created_at=datetime.utcnow(),
        status="pending",
        phase="phase1",
        user_profile=profile,
        preferences=payload.preferences,
        summary={
            "dry_run": bool(payload.dry_run),
            "skip_cv_approval": True,
            "skip_email_approval": True,
        },
    )
    db.add(run)
    await db.commit()

    task = asyncio.create_task(_execute_run(run_id, payload.dry_run))
    RUN_TASKS[run_id] = task

    return RunCreateResponse(run_id=run_id)


@router.post("/run/{run_id}/cancel")
async def cancel_run(run_id: str, db: AsyncSession = Depends(get_db)):
    run = await db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status not in ("running", "awaiting_cv_approval", "awaiting_email_approval"):
        return {"status": "ok", "message": f"Run is already {run.status}"}

    task = RUN_TASKS.get(run_id)
    if task and not task.done():
        task.cancel()

    run.status = "failed"
    run.summary = run.summary or {}
    run.summary["error"] = "Run was manually cancelled by the user."
    await db.commit()

    await _emit_step(run_id, "orchestrator", "RUN_CRASHED", {"error": "Run was manually cancelled by the user."}, "error")

    return {"status": "cancelled", "run_id": run_id}


@router.get("/run/{run_id}/stream")
async def run_stream(run_id: str):
    return await stream_run_events(run_id)


@router.get("/run/{run_id}")
async def get_run(run_id: str, db: AsyncSession = Depends(get_db)):
    run = await db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    jobs_result = await db.execute(select(Job).where(Job.run_id == run_id))
    jobs = jobs_result.scalars().all()

    sent_result = await db.execute(select(SentEmail).where(SentEmail.run_id == run_id))
    sent_emails = sent_result.scalars().all()

    hr_result = await db.execute(select(HREmail).where(HREmail.run_id == run_id))
    hr_emails = hr_result.scalars().all()

    # CV and Email approval counts
    cv_result = await db.execute(select(CVApproval).where(CVApproval.run_id == run_id))
    cv_approvals = cv_result.scalars().all()
    email_result = await db.execute(select(EmailApproval).where(EmailApproval.run_id == run_id))
    email_approvals = email_result.scalars().all()

    return {
        "id": run.id,
        "status": run.status,
        "phase": run.phase,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "user_profile": run.user_profile or {},
        "preferences": run.preferences or {},
        "summary": run.summary,
        "cv_approvals": {
            "total": len(cv_approvals),
            "pending": sum(1 for c in cv_approvals if c.status == "pending"),
            "approved": sum(1 for c in cv_approvals if c.status == "approved"),
            "rejected": sum(1 for c in cv_approvals if c.status == "rejected"),
        },
        "email_approvals": {
            "total": len(email_approvals),
            "pending": sum(1 for e in email_approvals if e.status == "pending"),
            "approved": sum(1 for e in email_approvals if e.status == "approved"),
            "rejected": sum(1 for e in email_approvals if e.status == "rejected"),
        },
        "jobs": [
            {
                "id": job.id,
                "job_id": _raw_job_id(job.id),
                "title": job.title,
                "company": job.company,
                "location": job.location,
                "description": job.description,
                "apply_url": job.apply_url,
                "source": job.source,
                "relevance_score": job.relevance_score,
            }
            for job in jobs
        ],
        "hr_emails": [
            {
                "job_id": _raw_job_id(email.job_id),
                "company": email.company,
                "email": email.email,
                "confidence": email.confidence,
                "source": email.source,
            }
            for email in hr_emails
        ],
        "emails": [
            {
                "job_id": _raw_job_id(email.job_id),
                "recipient": email.recipient,
                "subject": email.subject,
                "body": email.body,
                "status": email.status,
                "sent_at": email.sent_at.isoformat() if email.sent_at else None,
            }
            for email in sent_emails
        ],
    }


@router.get("/run/{run_id}/steps")
async def get_run_steps(
    run_id: str,
    limit: int = Query(default=1000, ge=1, le=5000),
    db: AsyncSession = Depends(get_db),
):
    run = await db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    result = await db.execute(
        select(Step)
        .where(Step.run_id == run_id)
        .order_by(Step.id.asc())
        .limit(limit)
    )
    steps = result.scalars().all()
    return {"run_id": run_id, "steps": [step.to_dict() for step in steps]}


@router.get("/runs")
async def list_runs(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Run).order_by(desc(Run.created_at)))
    runs = result.scalars().all()
    return {
        "runs": [
            {
                "id": run.id,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "status": run.status,
                "phase": run.phase,
                "summary": run.summary,
            }
            for run in runs
        ]
    }


@router.get("/auth/gmail")
async def gmail_auth_url():
    tool = GmailTool()
    return {"auth_url": tool.get_auth_url()}


@router.get("/auth/callback")
async def gmail_callback(code: str = Query(...)):
    tool = GmailTool()
    try:
        tokens = tool.exchange_code(code)
        token_path = Path(settings.gmail_tokens_path)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(json.dumps(tokens), encoding="utf-8")
        return RedirectResponse(url="http://localhost:3000/?gmail=connected")
    except Exception as exc:
        print(f"Gmail callback error: {exc}")
        return RedirectResponse(url="http://localhost:3000/?gmail=error")


@router.get("/auth/status")
async def auth_status():
    token_path = Path(settings.gmail_tokens_path)
    return {"gmail_connected": token_path.exists()}


@router.post("/auth/disconnect")
async def gmail_disconnect():
    token_path = Path(settings.gmail_tokens_path)
    try:
        if token_path.exists():
            token_path.unlink()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to disconnect Gmail: {exc}") from exc
    return {"gmail_connected": False}


# ═══════════════════════════════════════════════
# CV APPROVAL ENDPOINTS
# ═══════════════════════════════════════════════

@router.get("/run/{run_id}/cvs")
async def get_run_cvs(run_id: str, db: AsyncSession = Depends(get_db)):
    """Get all tailored CVs for a run with their approval status."""
    run = await db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    result = await db.execute(select(CVApproval).where(CVApproval.run_id == run_id))
    cvs = result.scalars().all()
    return {"run_id": run_id, "cvs": [cv.to_dict() for cv in cvs]}


@router.put("/run/{run_id}/cv/{job_id}/approve")
async def approve_cv(run_id: str, job_id: str, db: AsyncSession = Depends(get_db)):
    """Approve a single CV."""
    result = await db.execute(
        select(CVApproval).where(CVApproval.run_id == run_id, CVApproval.job_id == job_id)
    )
    cv = result.scalar_one_or_none()
    if not cv:
        raise HTTPException(status_code=404, detail="CV not found")

    cv.status = "approved"
    cv.updated_at = datetime.utcnow()
    await db.commit()
    return {"status": "approved", "job_id": job_id}


@router.put("/run/{run_id}/cv/{job_id}/reject")
async def reject_cv(run_id: str, job_id: str, db: AsyncSession = Depends(get_db)):
    """Reject a single CV (won't be used for email)."""
    result = await db.execute(
        select(CVApproval).where(CVApproval.run_id == run_id, CVApproval.job_id == job_id)
    )
    cv = result.scalar_one_or_none()
    if not cv:
        raise HTTPException(status_code=404, detail="CV not found")

    cv.status = "rejected"
    cv.updated_at = datetime.utcnow()
    await db.commit()
    return {"status": "rejected", "job_id": job_id}


@router.put("/run/{run_id}/cv/{job_id}/edit")
async def edit_cv(run_id: str, job_id: str, payload: CVEditRequest, db: AsyncSession = Depends(get_db)):
    """Save edited CV content."""
    result = await db.execute(
        select(CVApproval).where(CVApproval.run_id == run_id, CVApproval.job_id == job_id)
    )
    cv = result.scalar_one_or_none()
    if not cv:
        raise HTTPException(status_code=404, detail="CV not found")

    cv.edited_content = payload.content
    cv.updated_at = datetime.utcnow()
    await db.commit()
    return {"status": "saved", "job_id": job_id, "content": cv.to_dict()}


@router.post("/run/{run_id}/cv/{job_id}/chat")
async def chat_cv(run_id: str, job_id: str, payload: CVChatRequest, db: AsyncSession = Depends(get_db)):
    """Chat with AI to modify a specific CV."""
    result = await db.execute(
        select(CVApproval).where(CVApproval.run_id == run_id, CVApproval.job_id == job_id)
    )
    cv = result.scalar_one_or_none()
    if not cv:
        raise HTTPException(status_code=404, detail="CV not found")

    current = payload.current_content or cv.edited_content or cv.original_content

    prompt = (
        "You are an expert resume writer. Modify the CV sections based on the user's request.\n"
        "STRICT RULES:\n"
        "- NEVER fabricate experience, skills, companies, or education\n"
        "- ONLY rephrase, reorder, or add skills the candidate actually has\n"
        "- Keep the same JSON structure\n\n"
        f"USER REQUEST: {payload.message}\n\n"
        f"JOB DESCRIPTION:\n{payload.job_description[:3000]}\n\n"
        f"CURRENT CV CONTENT:\n{json.dumps(current, indent=2)}\n\n"
        "Return ONLY valid JSON with the MODIFIED cv content:\n"
        "{\n"
        '  "tailored_summary": "...",\n'
        '  "skills_reordered": ["..."],\n'
        '  "experience_bullets": {"Company": ["bullet1", ...]},\n'
        '  "keywords_matched": ["..."],\n'
        '  "match_score": 85\n'
        "}"
    )

    default = current
    if not gemini_enabled():
        return {"reply": "AI not available. Please edit manually.", "content": current}

    try:
        updated = await gemini_generate_json(
            prompt,
            default=default,
            temperature=0.2,
            max_output_tokens=1800,
            system_prompt="You are an expert resume writer. Modify the resume based on real information only.",
        )
        if isinstance(updated, dict):
            cv.edited_content = updated
            cv.updated_at = datetime.utcnow()
            await db.commit()
            return {"reply": "CV updated as requested.", "content": updated}
    except Exception as exc:
        return {"reply": f"Error: {exc}", "content": current}

    return {"reply": "Could not process request.", "content": current}


@router.post("/run/{run_id}/cvs/approve-all")
async def approve_all_cvs(run_id: str, db: AsyncSession = Depends(get_db)):
    """Approve all pending CVs and trigger Phase 2 (email drafting)."""
    run = await db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.phase != "awaiting_cv_approval":
        raise HTTPException(status_code=400, detail=f"Run is not awaiting CV approval (current phase: {run.phase})")

    # Approve all pending CVs
    result = await db.execute(
        select(CVApproval).where(CVApproval.run_id == run_id, CVApproval.status == "pending")
    )
    pending = result.scalars().all()
    for cv in pending:
        cv.status = "approved"
        cv.updated_at = datetime.utcnow()
    await db.commit()

    # Check if any CVs are approved
    approved_result = await db.execute(
        select(CVApproval).where(CVApproval.run_id == run_id, CVApproval.status == "approved")
    )
    approved_count = len(approved_result.scalars().all())

    if approved_count == 0:
        run.status = "done"
        run.phase = "done"
        summary = run.summary or {}
        summary["message"] = "All CVs rejected. No emails to send."
        run.summary = summary
        await db.commit()
        return {"status": "done", "message": "All CVs rejected. Pipeline complete."}

    # Start Phase 2
    task = asyncio.create_task(_execute_phase2(run_id))
    RUN_TASKS[run_id] = task

    return {"status": "phase2_started", "approved_count": approved_count}


# ═══════════════════════════════════════════════
# EMAIL APPROVAL ENDPOINTS
# ═══════════════════════════════════════════════

@router.get("/run/{run_id}/emails/drafts")
async def get_run_email_drafts(run_id: str, db: AsyncSession = Depends(get_db)):
    """Get all draft emails for a run with their approval status."""
    run = await db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    result = await db.execute(select(EmailApproval).where(EmailApproval.run_id == run_id))
    emails = result.scalars().all()

    # Recovery path for legacy/stuck runs:
    # awaiting_email_approval with no drafts should not block the pipeline forever.
    if run.phase == "awaiting_email_approval" and not emails:
        run.phase = "done"
        run.status = "done"
        summary = run.summary or {}
        summary["message"] = "No email drafts generated. Pipeline complete."
        summary["emails_drafted"] = 0
        summary["emails_processed"] = 0
        summary["emails_sent"] = 0
        run.summary = summary
        await db.commit()

    return {"run_id": run_id, "emails": [e.to_dict() for e in emails]}


@router.put("/run/{run_id}/email/{job_id}/approve")
async def approve_email(run_id: str, job_id: str, db: AsyncSession = Depends(get_db)):
    """Approve a single email."""
    result = await db.execute(
        select(EmailApproval).where(EmailApproval.run_id == run_id, EmailApproval.job_id == job_id)
    )
    email = result.scalar_one_or_none()
    if not email:
        raise HTTPException(status_code=404, detail="Email draft not found")

    email.status = "approved"
    email.updated_at = datetime.utcnow()
    await db.commit()
    return {"status": "approved", "job_id": job_id}


@router.put("/run/{run_id}/email/{job_id}/reject")
async def reject_email(run_id: str, job_id: str, db: AsyncSession = Depends(get_db)):
    """Reject a single email (won't be sent)."""
    result = await db.execute(
        select(EmailApproval).where(EmailApproval.run_id == run_id, EmailApproval.job_id == job_id)
    )
    email = result.scalar_one_or_none()
    if not email:
        raise HTTPException(status_code=404, detail="Email draft not found")

    email.status = "rejected"
    email.updated_at = datetime.utcnow()
    await db.commit()
    return {"status": "rejected", "job_id": job_id}


@router.put("/run/{run_id}/email/{job_id}/edit")
async def edit_email(run_id: str, job_id: str, payload: EmailEditRequest, db: AsyncSession = Depends(get_db)):
    """Save edited email subject/body."""
    result = await db.execute(
        select(EmailApproval).where(EmailApproval.run_id == run_id, EmailApproval.job_id == job_id)
    )
    email = result.scalar_one_or_none()
    if not email:
        raise HTTPException(status_code=404, detail="Email draft not found")

    if payload.subject:
        email.edited_subject = payload.subject
    if payload.body:
        email.edited_body = payload.body
    email.updated_at = datetime.utcnow()
    await db.commit()
    return {"status": "saved", "job_id": job_id, "email": email.to_dict()}


@router.post("/run/{run_id}/email/{job_id}/chat")
async def chat_email(run_id: str, job_id: str, payload: EmailChatRequest, db: AsyncSession = Depends(get_db)):
    """Chat with AI to modify an email draft."""
    result = await db.execute(
        select(EmailApproval).where(EmailApproval.run_id == run_id, EmailApproval.job_id == job_id)
    )
    email = result.scalar_one_or_none()
    if not email:
        raise HTTPException(status_code=404, detail="Email draft not found")

    current_subject = payload.current_subject or email.edited_subject or email.original_subject
    current_body = payload.current_body or email.edited_body or email.original_body

    prompt = (
        "Modify this job application email based on the user's request.\n"
        "RULES:\n"
        "- Keep it professional and under 200 words\n"
        "- DO NOT add phone numbers or email addresses\n"
        "- Keep the same format\n\n"
        f"USER REQUEST: {payload.message}\n\n"
        f"JOB: {payload.job_title} at {payload.company}\n\n"
        f"CURRENT SUBJECT: {current_subject}\n"
        f"CURRENT BODY:\n{current_body}\n\n"
        "Return ONLY valid JSON:\n"
        '{"subject": "...", "body": "..."}'
    )

    if not gemini_enabled():
        return {"reply": "AI not available. Please edit manually.", "subject": current_subject, "body": current_body}

    try:
        updated = await gemini_generate_json(
            prompt,
            default={"subject": current_subject, "body": current_body},
            temperature=0.2,
            max_output_tokens=600,
        )
        if isinstance(updated, dict):
            new_subject = updated.get("subject", current_subject)
            new_body = updated.get("body", current_body)
            email.edited_subject = new_subject
            email.edited_body = new_body
            email.updated_at = datetime.utcnow()
            await db.commit()
            return {"reply": "Email updated.", "subject": new_subject, "body": new_body}
    except Exception as exc:
        return {"reply": f"Error: {exc}", "subject": current_subject, "body": current_body}

    return {"reply": "Could not process.", "subject": current_subject, "body": current_body}


@router.post("/run/{run_id}/emails/approve-all")
async def approve_all_emails(run_id: str, db: AsyncSession = Depends(get_db)):
    """Approve all pending emails and trigger Phase 3 (send)."""
    run = await db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.phase != "awaiting_email_approval":
        raise HTTPException(status_code=400, detail=f"Run is not awaiting email approval (current phase: {run.phase})")

    # Approve all pending emails
    result = await db.execute(
        select(EmailApproval).where(EmailApproval.run_id == run_id, EmailApproval.status == "pending")
    )
    pending = result.scalars().all()
    for email in pending:
        email.status = "approved"
        email.updated_at = datetime.utcnow()
    await db.commit()

    # Check approved count
    approved_result = await db.execute(
        select(EmailApproval).where(EmailApproval.run_id == run_id, EmailApproval.status == "approved")
    )
    approved_count = len(approved_result.scalars().all())

    if approved_count == 0:
        run.status = "done"
        run.phase = "done"
        summary = run.summary or {}
        summary["message"] = "All emails rejected. Nothing to send."
        run.summary = summary
        await db.commit()
        return {"status": "done", "message": "All emails rejected. Pipeline complete."}

    # Start Phase 3
    task = asyncio.create_task(_execute_phase3(run_id))
    RUN_TASKS[run_id] = task

    return {"status": "phase3_started", "approved_count": approved_count}


# ═══════════════════════════════════════════════
# STANDALONE ENHANCE CV ENDPOINT (Multi-JD)
# ═══════════════════════════════════════════════

class BoostCVRequest(BaseModel):
    profile: dict = Field(default_factory=dict)
    jd_text: str = Field(min_length=1, max_length=10000)
    current_enhanced: dict = Field(default_factory=dict)
    target_score: int = 95
    cv_format: str = "ats"  # "ats" or "europe"


@router.post("/enhance-cv")
async def enhance_cv(
    resume: UploadFile = File(...),
    jd_texts: str = Form(default=""),
    jd_text: str = Form(default=""),
    jd_link: str = Form(default=""),
    db: AsyncSession = Depends(get_db),
):
    """Upload CV + multiple JDs → get match scores for each JD."""
    os.makedirs(settings.upload_dir, exist_ok=True)
    upload_path = Path(settings.upload_dir) / f"{uuid4()}_{resume.filename}"

    content = await resume.read()
    upload_path.write_bytes(content)

    try:
        parsed = await parse_resume(str(upload_path))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse resume: {exc}") from exc

    # Parse multiple JDs: try jd_texts (JSON array) first, then fall back to single jd_text
    jd_list: list[str] = []
    if jd_texts.strip():
        try:
            jd_list = json.loads(jd_texts)
            if isinstance(jd_list, str):
                jd_list = [jd_list]
        except json.JSONDecodeError:
            # Treat as single JD string
            jd_list = [jd_texts.strip()]
    elif jd_text.strip():
        jd_list = [jd_text.strip()]
    elif jd_link.strip():
        jd_list = [f"Job posting at: {jd_link.strip()}"]

    if not jd_list:
        raise HTTPException(status_code=400, detail="Please provide at least one job description.")

    # Filter empty strings
    jd_list = [jd.strip() for jd in jd_list if jd.strip()]
    if not jd_list:
        raise HTTPException(status_code=400, detail="All job descriptions are empty.")

    resume_text = parsed.get("raw_text", "")[:10000]

    # For each JD, get match score + enhanced content
    jd_results = []

    for idx, jd in enumerate(jd_list[:10]):  # Max 10 JDs
        system_prompt = (
            "You are the WORLD'S BEST ATS resume optimizer and career strategist. "
            "You have 20+ years of experience in recruitment, HR, and resume writing.\n\n"
            "YOUR MISSION: Analyze this resume against the job description and provide an accurate match score "
            "and enhanced CV content.\n\n"
            "WHAT YOU MUST DO (MANDATORY):\n"
            "1. COMPLETELY REWRITE the professional summary - make it laser-focused on this specific JD\n"
            "2. EXTRACT every single keyword, technology, tool, methodology from the JD\n"
            "3. INJECT those keywords into the resume naturally by rephrasing existing experience\n"
            "4. ADD implied/related skills the candidate actually has\n"
            "5. REORDER skills so JD-critical skills appear first\n"
            "6. REPHRASE every bullet point to use the EXACT terminology from the JD\n"
            "7. QUANTIFY achievements (added numbers, percentages, metrics)\n"
            "8. ADD action verbs that match the JD's language\n\n"
            "RULES:\n"
            "- NEVER invent companies, job titles, or education that don't exist\n"
            "- NEVER change employment dates\n"
            "- You CAN expand bullet points with more detail from the candidate's likely experience\n"
            "- You CAN add skills the candidate probably has based on their tech stack\n"
            "- You CAN rewrite summaries and bullet points completely\n"
            "- Make the match_score REALISTIC - be HONEST about the match percentage\n"
            "- ALWAYS provide at least 12-20 skills\n"
            "- ALWAYS provide at least 3-5 bullet points per company\n"
            "- ALWAYS list at least 8-15 matched keywords"
        )

        user_prompt = (
            "TASK: Analyze and enhance this resume for the job description below.\n\n"
            "===== JOB DESCRIPTION (ANALYZE EVERY WORD) =====\n"
            f"{jd[:6000]}\n\n"
            "===== ORIGINAL RESUME (SOURCE MATERIAL) =====\n"
            f"{resume_text}\n\n"
            "===== YOUR INSTRUCTIONS =====\n"
            "1. Read the JD word by word. Extract EVERY skill, technology, qualification, and requirement.\n"
            "2. For EACH requirement, find or create a match in the resume.\n"
            "3. Rewrite the summary: it MUST mention the top 3-5 JD requirements directly.\n"
            "4. Reorder skills: JD skills FIRST, then supporting skills.\n"
            "5. Rewrite EVERY experience bullet: use JD keywords and action verbs.\n"
            "6. List EVERY keyword you matched between JD and resume.\n"
            "7. Calculate match_score: what %% of JD requirements are now covered?\n\n"
            "CRITICAL: The enhanced CV MUST be SIGNIFICANTLY DIFFERENT from the original. "
            "If you return the original content unchanged, you have FAILED.\n\n"
            "Return ONLY valid JSON (no markdown, no backticks, no explanation):\n"
            "{\n"
            '  "tailored_summary": "3-4 sentence professional summary addressing top JD requirements",\n'
            '  "skills_reordered": ["JD skill 1", "JD skill 2", ... at least 12-20 skills],\n'
            '  "experience_bullets": {\n'
            '    "Company Name": ["bullet with JD keyword", ...at least 3-5 bullets per company]\n'
            '  },\n'
            '  "keywords_matched": ["keyword1", "keyword2", ... list every matched keyword],\n'
            '  "match_score": 75\n'
            "}"
        )

        default = {
            "tailored_summary": parsed.get("summary", ""),
            "skills_reordered": parsed.get("skills", []),
            "experience_bullets": {exp.get("company", "Exp"): exp.get("bullets", []) for exp in parsed.get("experience", [])[:5]},
            "keywords_matched": [],
            "match_score": 40,
        }

        enhanced = default
        if gemini_enabled():
            try:
                enhanced = await gemini_generate_json(
                    user_prompt,
                    default=default,
                    system_prompt=system_prompt,
                    temperature=0.4,
                    max_output_tokens=8192,
                )
                if not isinstance(enhanced, dict):
                    enhanced = default

                for key in ["tailored_summary", "skills_reordered", "experience_bullets", "keywords_matched", "match_score"]:
                    if key not in enhanced:
                        enhanced[key] = default.get(key, "")

                try:
                    enhanced["match_score"] = int(float(enhanced.get("match_score", 40)))
                except (ValueError, TypeError):
                    enhanced["match_score"] = 40

                if not enhanced.get("skills_reordered") or len(enhanced["skills_reordered"]) < 3:
                    enhanced["skills_reordered"] = default.get("skills_reordered", []) or ["See original resume"]

                if not enhanced.get("experience_bullets") or not isinstance(enhanced["experience_bullets"], dict):
                    enhanced["experience_bullets"] = default.get("experience_bullets", {})

            except Exception as exc:
                print(f"Enhance CV error for JD #{idx}: {exc}")
                enhanced = default

        jd_snippet = jd[:150].strip().replace("\n", " ")
        jd_results.append({
            "jd_index": idx,
            "jd_snippet": jd_snippet,
            "jd_full": jd,
            "match_score": enhanced.get("match_score", 40),
            "enhanced_cv": enhanced,
        })

    return {
        "profile": parsed,
        "jd_results": jd_results,
        "session_id": str(uuid4()),
        "message": f"Analyzed {len(jd_results)} job description(s).",
    }


@router.post("/enhance-cv/boost")
async def boost_cv_match(payload: BoostCVRequest):
    """Boost CV match score to target (95%) for a specific JD and format."""
    profile = payload.profile
    jd = payload.jd_text
    current = payload.current_enhanced
    target = payload.target_score
    cv_format = payload.cv_format  # "ats" or "europe"

    resume_text = profile.get("raw_text", "")[:10000]

    format_instruction = ""
    if cv_format == "europe":
        format_instruction = (
            "\n\nADDITIONAL EUROPASS/EUROPE CV FORMAT REQUIREMENTS:\n"
            "- Include a 'personal_info' field with: full_name, email, phone, address, date_of_birth, nationality, linkedin\n"
            "- Include a 'languages' field: [{\"language\": \"...\", \"level\": \"Native/C2/C1/B2/B1/A2/A1\"}]\n"
            "- Include a 'certifications' field: [\"cert1\", \"cert2\"]\n"
            "- Experience bullets should include dates in European format\n"
        )

    system_prompt = (
        "You are the WORLD'S #1 ATS resume optimizer. Your SOLE MISSION is to achieve "
        f"a {target}%+ ATS match score.\n\n"
        "YOU MUST BE AGGRESSIVE:\n"
        "1. Extract EVERY keyword from the JD\n"
        "2. INJECT all missing keywords into the resume naturally\n"
        "3. REWRITE the summary to mirror the JD requirements\n"
        "4. ADD all related/implied skills\n"
        "5. REPHRASE every bullet to match JD's language\n"
        "6. ADD metrics and quantified results to every bullet\n"
        "7. MAXIMIZE keyword density without being unnatural\n\n"
        "CRITICAL: You MUST include ALL data from the original resume:\n"
        "- Every company, title, and date from work experience\n"
        "- Every degree, institution, and date from education\n"
        "- Every project with description\n"
        "- All contact info, links (portfolio, github, linkedin)\n"
        "- All courses and certifications\n\n"
        "RULES:\n"
        "- NEVER fabricate companies, job titles, or education\n"
        "- NEVER drop any experience, education, or project from the original CV\n"
        "- You CAN add skills and rewrite bullets\n"
        f"- TARGET: {target}%+ match score\n"
        "- At least 15-25 skills, 3-5 bullets per company, 10-20 matched keywords"
        f"{format_instruction}"
    )

    json_structure = (
        "{\n"
        '  "tailored_summary": "powerful summary with ALL top JD keywords",\n'
        '  "skills_reordered": ["skill1", "skill2", ... 15-25 skills],\n'
        '  "experience_detailed": [\n'
        '    {"company": "Company Name", "title": "Job Title", "duration": "Start - End", "location": "City, Country", "bullets": ["bullet1", "bullet2", ...]}\n'
        '  ],\n'
        '  "education": [\n'
        '    {"degree": "Degree Name", "institution": "School", "duration": "Start - End", "details": "GPA/percentage if available"}\n'
        '  ],\n'
        '  "projects": [\n'
        '    {"name": "Project Name", "description": "what it does and what tech used"}\n'
        '  ],\n'
        '  "courses": [\n'
        '    {"name": "Course Name", "institution": "Where", "duration": "Start - End"}\n'
        '  ],\n'
        '  "keywords_matched": ["kw1", "kw2", ... 10-20 keywords],\n'
        f'  "match_score": {target}\n'
    )
    if cv_format == "europe":
        json_structure += (
            '  "personal_info": {"full_name": "...", "email": "...", "phone": "...", "address": "...", '
            '"date_of_birth": "...", "nationality": "...", "linkedin": "..."},\n'
            '  "languages": [{"language": "...", "level": "..."}],\n'
            '  "certifications": ["cert1", "cert2"]\n'
        )
    json_structure += "}"

    user_prompt = (
        f"MISSION: Boost this CV to {target}%+ match with this specific JD.\n\n"
        "CRITICAL RULES:\n"
        "1. Include ALL experience from the original resume with EXACT company names, job titles, and dates\n"
        "2. Include ALL education with degrees, institutions, dates, and GPA/marks\n"
        "3. Include ALL projects from the original resume\n"
        "4. Include ALL courses and certifications\n"
        "5. Do NOT skip or miss any section from the original resume\n\n"
        "===== JOB DESCRIPTION =====\n"
        f"{jd[:6000]}\n\n"
        "===== ORIGINAL RESUME (INCLUDE ALL DATA FROM THIS) =====\n"
        f"{resume_text}\n\n"
        f"IMPROVE everything to reach {target}%+ match while keeping ALL original data.\n\n"
        f"Return ONLY valid JSON:\n{json_structure}"
    )

    default = current or {
        "tailored_summary": profile.get("summary", ""),
        "skills_reordered": profile.get("skills", []),
        "experience_detailed": [],
        "education": [],
        "projects": [],
        "courses": [],
        "keywords_matched": [],
        "match_score": 50,
    }

    if not gemini_enabled():
        return {"enhanced_cv": default, "message": "AI not available.", "match_score": default.get("match_score", 50)}

    try:
        boosted = await gemini_generate_json(
            user_prompt,
            default=default,
            system_prompt=system_prompt,
            temperature=0.5,
            max_output_tokens=12000,
        )
        if not isinstance(boosted, dict):
            boosted = default

        for key in ["tailored_summary", "skills_reordered", "experience_detailed", "education", "projects", "courses", "keywords_matched", "match_score"]:
            if key not in boosted:
                boosted[key] = default.get(key, [] if key in ["skills_reordered", "experience_detailed", "education", "projects", "courses", "keywords_matched"] else "")

        # Backward compat: if AI returned experience_bullets instead of experience_detailed, convert
        if not boosted.get("experience_detailed") and boosted.get("experience_bullets"):
            exp_b = boosted["experience_bullets"]
            if isinstance(exp_b, dict):
                boosted["experience_detailed"] = [
                    {"company": comp, "title": "", "duration": "", "location": "", "bullets": buls if isinstance(buls, list) else []}
                    for comp, buls in exp_b.items()
                ]

        try:
            boosted["match_score"] = int(float(boosted.get("match_score", 50)))
        except (ValueError, TypeError):
            boosted["match_score"] = 50

    except Exception as exc:
        print(f"Boost CV error: {exc}")
        boosted = default

    return {
        "enhanced_cv": boosted,
        "match_score": boosted.get("match_score", 50),
        "format": cv_format,
        "message": f"CV boosted to {boosted.get('match_score', 50)}% match ({cv_format.upper()} format).",
    }


@router.post("/enhance-cv/chat")
async def enhance_cv_chat(payload: CVChatRequest):
    """Follow-up chat to modify the enhanced CV."""
    current = payload.current_content

    prompt = (
        "You are the WORLD'S BEST ATS resume optimizer. The user wants to modify their enhanced CV.\n\n"
        "IMPORTANT: When the user asks to increase the match score, you MUST:\n"
        "1. Find MORE keywords from the JD that can be naturally incorporated\n"
        "2. Rewrite the summary to be even MORE targeted and specific\n"
        "3. Add MORE relevant skills from the JD (skills the candidate likely has)\n"
        "4. Rephrase experience bullets to use MORE JD terminology and add metrics\n"
        "5. Actually INCREASE the match_score number in your response\n"
        "6. Add MORE keywords to the keywords_matched list\n\n"
        "STRICT RULES:\n"
        "- NEVER fabricate experience or companies\n"
        "- You CAN add skills the candidate likely has based on their experience\n"
        "- You CAN rephrase bullets to match JD terminology\n"
        "- You CAN expand bullet points with more detail\n"
        "- Be AGGRESSIVE in optimization - the user wants MAXIMUM ATS matching\n\n"
        f"USER REQUEST: {payload.message}\n\n"
        f"JOB DESCRIPTION:\n{payload.job_description[:5000]}\n\n"
        f"CURRENT CV CONTENT:\n{json.dumps(current, indent=2)}\n\n"
        "Return ONLY valid JSON with the IMPROVED content (MUST be different from input):\n"
        "{\n"
        '  "tailored_summary": "improved summary with more JD keywords",\n'
        '  "skills_reordered": ["skill1", "skill2", ... more skills],\n'
        '  "experience_bullets": {"Company": ["improved bullet1 with metrics", ...]},\n'
        '  "keywords_matched": ["more keywords from JD"],\n'
        '  "match_score": <higher score reflecting improvements>\n'
        "}"
    )

    if not gemini_enabled():
        return {"reply": "AI not available. Please edit manually.", "content": current}

    try:
        updated = await gemini_generate_json(
            prompt,
            default=current,
            temperature=0.4,
            max_output_tokens=8192,
            system_prompt="You are an expert ATS resume optimizer. ALWAYS make significant improvements. Never return the same content.",
        )
        if isinstance(updated, dict):
            new_score = updated.get("match_score", 0)
            return {"reply": f"CV updated! Match score: {new_score}%", "content": updated}
    except Exception as exc:
        return {"reply": f"Error: {exc}", "content": current}

    return {"reply": "Could not process request.", "content": current}


@router.post("/enhance-cv/download")
async def download_enhanced_cv(
    payload: dict = None,
):
    """Generate and download a PDF from enhanced CV data. Template 1=ATS, 2=Europe."""
    if not payload:
        raise HTTPException(status_code=400, detail="No CV data provided.")

    from reportlab.lib.pagesizes import LETTER, A4
    from reportlab.lib.colors import HexColor
    from reportlab.pdfgen import canvas as pdf_canvas
    import textwrap

    enhanced = payload.get("enhanced_cv", {})
    profile = payload.get("profile", {})
    template = int(payload.get("template", 1))

    os.makedirs("uploads/enhanced", exist_ok=True)
    pdf_path = str(Path("uploads/enhanced") / f"enhanced_cv_{uuid4().hex[:8]}.pdf")

    page_size = A4 if template == 2 else LETTER
    c = pdf_canvas.Canvas(pdf_path, pagesize=page_size)
    width, height = page_size

    name = profile.get("full_name", "Candidate")
    email_addr = profile.get("email", "")
    phone = profile.get("phone", "")
    location = profile.get("location", "")
    linkedin = profile.get("linkedin_url", "")
    portfolio = profile.get("portfolio_url", "")
    github = profile.get("github_url", "")
    contact_parts = [v for v in [email_addr, phone, location] if v]
    link_parts = [v for v in [linkedin, portfolio, github] if v]
    summary_text = enhanced.get("tailored_summary", "")
    skills = enhanced.get("skills_reordered", [])
    # New detailed structure
    exp_detailed = enhanced.get("experience_detailed", [])
    edu_list = enhanced.get("education", []) or profile.get("education", [])
    projects = enhanced.get("projects", [])
    courses = enhanced.get("courses", [])
    # Backward compat: convert old experience_bullets to experience_detailed
    if not exp_detailed and enhanced.get("experience_bullets"):
        eb = enhanced["experience_bullets"]
        if isinstance(eb, dict):
            exp_detailed = [
                {"company": comp, "title": "", "duration": "", "location": "", "bullets": buls if isinstance(buls, list) else []}
                for comp, buls in eb.items()
            ]

    # Europe-specific fields
    personal_info = enhanced.get("personal_info", {})
    languages = enhanced.get("languages", [])
    certifications = enhanced.get("certifications", [])

    # ── Helper: wrapped text with page break ──
    def _wrap(cobj, text, x, y, font="Helvetica", size=10, wchars=95, lh=13, color=None):
        cobj.setFont(font, size)
        if color:
            cobj.setFillColor(color)
        for line in textwrap.wrap(str(text), width=wchars) or [""]:
            if y < 55:
                cobj.showPage()
                y = height - 50
                cobj.setFont(font, size)
                if color:
                    cobj.setFillColor(color)
            cobj.drawString(x, y, line)
            y -= lh
        if color:
            cobj.setFillColor(HexColor("#000000"))
        return y

    def _section_title_ats(cobj, title, x, y_pos):
        if y_pos < 80:
            cobj.showPage()
            y_pos = height - 50
        cobj.setFont("Helvetica-Bold", 11)
        cobj.drawString(x, y_pos, title)
        y_pos -= 5
        cobj.setStrokeColor(HexColor("#888888"))
        cobj.setLineWidth(0.4)
        cobj.line(x, y_pos, width - 50, y_pos)
        return y_pos - 14

    # ══════════════════════════════════════════════
    #  TEMPLATE 1: ATS-FRIENDLY (clean, keyword-optimized)
    # ══════════════════════════════════════════════
    if template == 1:
        x, y = 50, height - 45

        # Name
        c.setFont("Helvetica-Bold", 20)
        c.drawString(x, y, name.upper())
        y -= 20

        # Contact line
        c.setFont("Helvetica", 9)
        c.setFillColor(HexColor("#555555"))
        if contact_parts:
            c.drawString(x, y, "  |  ".join(contact_parts))
            y -= 13
        if link_parts:
            c.drawString(x, y, "  |  ".join(link_parts))
            y -= 13
        c.setFillColor(HexColor("#000000"))

        # Divider
        y -= 3
        c.setStrokeColor(HexColor("#222222"))
        c.setLineWidth(1.5)
        c.line(x, y, width - 50, y)
        y -= 16

        # Professional Summary
        if summary_text:
            y = _section_title_ats(c, "PROFESSIONAL SUMMARY", x, y)
            y = _wrap(c, summary_text, x, y, "Helvetica", 10, 95, 13)
            y -= 12

        # Technical Skills
        if skills:
            y = _section_title_ats(c, "TECHNICAL SKILLS", x, y)
            y = _wrap(c, "  •  ".join(skills), x, y, "Helvetica", 10, 95, 13)
            y -= 12

        # Experience (detailed with title/date/location)
        if exp_detailed:
            y = _section_title_ats(c, "PROFESSIONAL EXPERIENCE", x, y)
            for exp_item in exp_detailed:
                if y < 80:
                    c.showPage()
                    y = height - 50
                comp = exp_item.get("company", "")
                title = exp_item.get("title", "")
                duration = exp_item.get("duration", "")
                loc = exp_item.get("location", "")
                bullets = exp_item.get("bullets", [])

                # Company + Title line
                header = title
                if comp:
                    header = f"{title} — {comp}" if title else comp
                c.setFont("Helvetica-Bold", 10)
                c.drawString(x, y, header)
                # Duration + Location on right
                right_text = ""
                if duration:
                    right_text = duration
                if loc:
                    right_text += f"  |  {loc}" if right_text else loc
                if right_text:
                    c.setFont("Helvetica", 9)
                    c.setFillColor(HexColor("#555555"))
                    c.drawRightString(width - 50, y, right_text)
                    c.setFillColor(HexColor("#000000"))
                y -= 14

                for bul in (bullets if isinstance(bullets, list) else [])[:6]:
                    y = _wrap(c, f"•  {bul}", x + 14, y, "Helvetica", 10, 85, 13)
                y -= 8

        # Education
        if edu_list:
            y = _section_title_ats(c, "EDUCATION", x, y)
            for edu in edu_list[:6]:
                if y < 55:
                    c.showPage()
                    y = height - 50
                deg = edu.get("degree", "")
                inst = edu.get("institution", "")
                dur = edu.get("duration", "") or edu.get("year", "")
                details = edu.get("details", "")

                header = f"{deg} — {inst}" if inst else deg
                c.setFont("Helvetica-Bold", 10)
                c.drawString(x, y, header)
                if dur:
                    c.setFont("Helvetica", 9)
                    c.setFillColor(HexColor("#555555"))
                    c.drawRightString(width - 50, y, str(dur))
                    c.setFillColor(HexColor("#000000"))
                y -= 14
                if details:
                    y = _wrap(c, details, x + 14, y, "Helvetica", 9, 88, 12, color=HexColor("#444444"))
                y -= 4

        # Projects
        if projects:
            y = _section_title_ats(c, "PROJECTS", x, y)
            for proj in projects[:6]:
                if y < 55:
                    c.showPage()
                    y = height - 50
                pname = proj.get("name", "") if isinstance(proj, dict) else str(proj)
                pdesc = proj.get("description", "") if isinstance(proj, dict) else ""
                c.setFont("Helvetica-Bold", 10)
                c.drawString(x, y, pname)
                y -= 13
                if pdesc:
                    y = _wrap(c, pdesc, x + 14, y, "Helvetica", 9, 88, 12, color=HexColor("#444444"))
                y -= 6

        # Courses
        if courses:
            y = _section_title_ats(c, "COURSES & CERTIFICATIONS", x, y)
            for course in courses[:8]:
                if y < 55:
                    c.showPage()
                    y = height - 50
                cname = course.get("name", "") if isinstance(course, dict) else str(course)
                cinst = course.get("institution", "") if isinstance(course, dict) else ""
                cdur = course.get("duration", "") if isinstance(course, dict) else ""
                line = cname
                if cinst:
                    line += f" — {cinst}"
                c.setFont("Helvetica", 10)
                c.drawString(x, y, f"•  {line}")
                if cdur:
                    c.setFont("Helvetica", 8)
                    c.setFillColor(HexColor("#555555"))
                    c.drawRightString(width - 50, y, cdur)
                    c.setFillColor(HexColor("#000000"))
                y -= 13

    # ══════════════════════════════════════════════
    #  TEMPLATE 2: EUROPE CV (Europass-inspired)
    # ══════════════════════════════════════════════
    elif template == 2:
        accent = HexColor("#1B4F72")
        light_accent = HexColor("#2980B9")
        gray = HexColor("#555555")
        black = HexColor("#000000")

        x, y = 50, height - 45
        right_col = 180

        # ── Header bar ──
        c.setFillColor(accent)
        c.rect(0, height - 80, width, 80, fill=True, stroke=False)
        c.setFillColor(HexColor("#FFFFFF"))
        c.setFont("Helvetica-Bold", 22)
        c.drawString(50, height - 35, name.upper())
        c.setFont("Helvetica", 10)
        pi_title = personal_info.get("title", "")
        if pi_title:
            c.drawString(50, height - 52, pi_title)
        if contact_parts:
            c.drawString(50, height - 67, "  |  ".join(contact_parts))
        c.setFillColor(black)

        y = height - 105

        # ── Personal Information (left column style) ──
        def _section_header(cobj, title, x_pos, y_pos):
            cobj.setFont("Helvetica-Bold", 11)
            cobj.setFillColor(accent)
            cobj.drawString(x_pos, y_pos, title.upper())
            y_pos -= 4
            cobj.setStrokeColor(light_accent)
            cobj.setLineWidth(1.2)
            cobj.line(x_pos, y_pos, x_pos + 120, y_pos)
            cobj.setFillColor(black)
            return y_pos - 14

        def _label_value(cobj, label, value, x_pos, y_pos, lh=14):
            if not value:
                return y_pos
            cobj.setFont("Helvetica-Bold", 8)
            cobj.setFillColor(gray)
            cobj.drawString(x_pos, y_pos, label.upper())
            cobj.setFont("Helvetica", 9)
            cobj.setFillColor(black)
            cobj.drawString(x_pos + 80, y_pos, str(value))
            return y_pos - lh

        # Personal Details section
        y = _section_header(c, "Personal Details", x, y)
        pi = personal_info or {}
        y = _label_value(c, "Full Name", pi.get("full_name", name), x, y)
        y = _label_value(c, "Email", pi.get("email", email_addr), x, y)
        y = _label_value(c, "Phone", pi.get("phone", phone), x, y)
        y = _label_value(c, "Address", pi.get("address", location), x, y)
        y = _label_value(c, "Nationality", pi.get("nationality", ""), x, y)
        y = _label_value(c, "Date of Birth", pi.get("date_of_birth", ""), x, y)
        y = _label_value(c, "LinkedIn", pi.get("linkedin", linkedin), x, y)
        y -= 10

        # Professional Summary
        if summary_text:
            y = _section_header(c, "Professional Profile", x, y)
            y = _wrap(c, summary_text, x, y, "Helvetica", 9.5, 90, 13)
            y -= 10

        # Skills
        if skills:
            y = _section_header(c, "Skills & Competencies", x, y)
            # Display as 2-column grid
            col_width = (width - 100) // 2
            for i in range(0, len(skills), 2):
                if y < 55:
                    c.showPage()
                    y = height - 50
                c.setFont("Helvetica", 9)
                c.drawString(x, y, f"•  {skills[i]}")
                if i + 1 < len(skills):
                    c.drawString(x + col_width, y, f"•  {skills[i + 1]}")
                y -= 13
            y -= 8

        # Work Experience (detailed)
        if exp_detailed:
            y = _section_header(c, "Work Experience", x, y)
            for exp_item in exp_detailed:
                if y < 80:
                    c.showPage()
                    y = height - 50
                comp = exp_item.get("company", "")
                title = exp_item.get("title", "")
                duration = exp_item.get("duration", "")
                loc = exp_item.get("location", "")
                bullets = exp_item.get("bullets", [])

                header = title
                if comp:
                    header = f"{title} — {comp}" if title else comp
                c.setFont("Helvetica-Bold", 10)
                c.setFillColor(accent)
                c.drawString(x, y, header)
                c.setFillColor(black)
                right_text = ""
                if duration:
                    right_text = duration
                if loc:
                    right_text += f"  |  {loc}" if right_text else loc
                if right_text:
                    c.setFont("Helvetica", 8)
                    c.setFillColor(gray)
                    c.drawRightString(width - 50, y, right_text)
                    c.setFillColor(black)
                y -= 14
                for bul in (bullets if isinstance(bullets, list) else [])[:6]:
                    y = _wrap(c, f"•  {bul}", x + 10, y, "Helvetica", 9, 88, 12)
                y -= 8

        # Education
        if edu_list:
            y = _section_header(c, "Education", x, y)
            for edu in edu_list[:6]:
                if y < 55:
                    c.showPage()
                    y = height - 50
                deg = edu.get("degree", "")
                inst = edu.get("institution", "")
                dur = edu.get("duration", "") or edu.get("year", "")
                details = edu.get("details", "")
                c.setFont("Helvetica-Bold", 9)
                c.setFillColor(accent)
                c.drawString(x, y, deg)
                c.setFillColor(black)
                c.setFont("Helvetica", 9)
                if inst:
                    c.drawString(x + 220, y, inst)
                if dur:
                    c.drawRightString(width - 50, y, str(dur))
                y -= 14
                if details:
                    y = _wrap(c, details, x + 10, y, "Helvetica", 8.5, 88, 11, color=gray)
                y -= 4

        # Projects
        if projects:
            y = _section_header(c, "Projects", x, y)
            for proj in projects[:6]:
                if y < 55:
                    c.showPage()
                    y = height - 50
                pname = proj.get("name", "") if isinstance(proj, dict) else str(proj)
                pdesc = proj.get("description", "") if isinstance(proj, dict) else ""
                c.setFont("Helvetica-Bold", 9)
                c.setFillColor(accent)
                c.drawString(x, y, pname)
                c.setFillColor(black)
                y -= 12
                if pdesc:
                    y = _wrap(c, pdesc, x + 10, y, "Helvetica", 8.5, 88, 11, color=gray)
                y -= 6

        # Courses
        if courses:
            y = _section_header(c, "Courses & Training", x, y)
            for course in courses[:8]:
                if y < 55:
                    c.showPage()
                    y = height - 50
                cname = course.get("name", "") if isinstance(course, dict) else str(course)
                cinst = course.get("institution", "") if isinstance(course, dict) else ""
                cdur = course.get("duration", "") if isinstance(course, dict) else ""
                line = cname
                if cinst:
                    line += f" — {cinst}"
                c.setFont("Helvetica", 9)
                c.drawString(x, y, f"•  {line}")
                if cdur:
                    c.setFont("Helvetica", 8)
                    c.setFillColor(gray)
                    c.drawRightString(width - 50, y, cdur)
                    c.setFillColor(black)
                y -= 13

        # Languages
        if languages:
            y -= 4
            y = _section_header(c, "Languages", x, y)
            for lang in languages[:8]:
                if y < 55:
                    c.showPage()
                    y = height - 50
                lang_name = lang.get("language", "") if isinstance(lang, dict) else str(lang)
                lang_level = lang.get("level", "") if isinstance(lang, dict) else ""
                c.setFont("Helvetica", 9)
                c.drawString(x, y, f"•  {lang_name}")
                if lang_level:
                    c.setFont("Helvetica-Oblique", 8)
                    c.setFillColor(gray)
                    c.drawString(x + 150, y, f"({lang_level})")
                    c.setFillColor(black)
                y -= 13

        # Certifications
        if certifications:
            y -= 4
            y = _section_header(c, "Certifications", x, y)
            for cert in certifications[:6]:
                if y < 55:
                    c.showPage()
                    y = height - 50
                c.setFont("Helvetica", 9)
                c.drawString(x, y, f"•  {cert}")
                y -= 13

    c.save()
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"enhanced_cv_{template}.pdf",
    )
