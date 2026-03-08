import asyncio
import logging
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.orchestrator import AgentState
from agents.job_search_agent import (
    _compact_query,
    _dedupe_jobs,
    _pick_three_queries,
    get_linkedin_provider_reason,
    score_job_against_cv,
    search_linkedin_jobs,
)
from api.routes import RUN_TASKS, _save_profile
from models.database import Job, Run, get_db, async_session
from tools.gemini_tool import gemini_generate_json, gemini_enabled
from tools.resume_parser import parse_resume

router = APIRouter()
logger = logging.getLogger(__name__)

class CVSearchRequest(BaseModel):
    run_id: str
    job_title: str
    location: str
    remote_only: bool
    skills: list[str] = Field(default_factory=list)
    profile_data: dict = Field(default_factory=dict)
    max_results: int = 15


class CVCreateRunRequest(BaseModel):
    profile: dict = Field(default_factory=dict)
    preferences: dict = Field(default_factory=dict)
    dry_run: bool = True


class CVStartRunRequest(BaseModel):
    run_id: str | None = None
    selected_jobs: list[dict]
    profile: dict
    preferences: dict
    dry_run: bool = False

import os


def _normalize_selected_jobs(selected_jobs: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    seen_job_ids: set[str] = set()

    for idx, raw_job in enumerate(selected_jobs):
        if not isinstance(raw_job, dict):
            continue

        job = dict(raw_job)
        candidate_id = str(job.get("job_id") or job.get("id") or job.get("link") or "").strip()
        if not candidate_id:
            candidate_id = f"selected_{idx}_{uuid4().hex[:8]}"

        dedupe_id = candidate_id
        suffix = 1
        while dedupe_id in seen_job_ids:
            dedupe_id = f"{candidate_id}_{suffix}"
            suffix += 1
        seen_job_ids.add(dedupe_id)

        job["job_id"] = dedupe_id
        job.setdefault("id", dedupe_id)
        normalized.append(job)

    return normalized


@router.post("/parse")
async def parse_cv_for_intake(
    resume: UploadFile = File(...),
):
    """Parses a CV file and extracts standard profile + suggested job title."""
    try:
        content = await resume.read()
        file_ext = os.path.splitext(resume.filename)[1]
        unique_filename = f"{uuid4()}{file_ext}"
        upload_dir = Path("uploads") / "resumes"
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / unique_filename
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read/save file: {e}")

    parsed = await parse_resume(str(file_path))
    if not parsed or "error" in parsed:
        raise HTTPException(status_code=400, detail=parsed.get("error", "Failed to parse resume"))

    # Extract suggested job title (latest role in experience)
    suggested_title = "Software Engineer"
    exp = parsed.get("experience", [])
    if exp and len(exp) > 0:
        first_role = exp[0].get("title", "")
        if first_role:
            suggested_title = first_role

    # Use LLM to cleanly format the title if Gemini is enabled
    if gemini_enabled() and suggested_title != "Software Engineer":
        try:
            clean_prompt = f"Extract a clean, standard job title from this raw title: '{suggested_title}'. Return ONLY JSON: {{'title': 'clean title'}}"
            res = await gemini_generate_json(clean_prompt, {"title": suggested_title})
            suggested_title = res.get("title", suggested_title)
        except:
            pass
            
    return {
        "profile": parsed,
        "suggested_job_title": suggested_title
    }

async def _resolve_running_conflict(db: AsyncSession) -> None:
    running_result = await db.execute(select(Run).where(Run.status == "running").limit(1))
    running = running_result.scalar_one_or_none()
    if not running:
        return

    task = RUN_TASKS.get(running.id)
    if not task or task.done():
        running.status = "failed"
        running.summary = {"error": "Run was interrupted by server restart or silent crash."}
        await db.commit()
        return

    raise HTTPException(
        status_code=409,
        detail=f"Another run is already in progress ({running.id}). Wait for completion before starting a new run.",
    )


@router.post("/create-run")
async def create_cv_intake_run(payload: CVCreateRunRequest, db: AsyncSession = Depends(get_db)):
    """Create an intake run immediately so session history + trace can bind to a real run id."""
    await _resolve_running_conflict(db)

    profile_data = payload.profile if isinstance(payload.profile, dict) else {}
    preferences = payload.preferences if isinstance(payload.preferences, dict) else {}
    await _save_profile(db, profile_data)

    run_id = str(uuid4())
    run = Run(
        id=run_id,
        created_at=datetime.utcnow(),
        status="pending",
        phase="phase1",
        user_profile=profile_data,
        preferences=preferences,
        summary={
            "dry_run": bool(payload.dry_run),
            "intake_stage": "initialized",
            "skip_cv_approval": True,
            "skip_email_approval": True,
        },
    )
    db.add(run)
    await db.commit()

    return {"run_id": run_id}


@router.post("/search-jobs")
async def search_jobs_for_cv(payload: CVSearchRequest, db: AsyncSession = Depends(get_db)):
    """Searches jobs for an existing intake run and records full trace under that run id."""
    run = await db.get(Run, payload.run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found. Create intake run first.")

    run.status = "running"
    run.phase = "phase1"
    summary = run.summary or {}
    summary["intake_stage"] = "job_search"
    summary["dry_run"] = bool(summary.get("dry_run", True))
    run.summary = summary
    await db.commit()

    async def emit(agent: str, event_type: str, data: dict, status: str = "running") -> None:
        await _emit_step(payload.run_id, agent, event_type, data, status)

    await emit(
        "orchestrator",
        "RUN_STARTED",
        {"run_id": payload.run_id, "phase": "phase1", "timestamp": datetime.utcnow().isoformat()},
        "running",
    )
    await emit("orchestrator", "AGENT_START", {"agent": "job_search"}, "running")
    await emit(
        "job_search",
        "SEARCHING_LINKEDIN",
        {
            "title": payload.job_title,
            "location": payload.location,
            "mode": "cv_intake",
        },
        "running",
    )

    profile = payload.profile_data if isinstance(payload.profile_data, dict) else {}
    fallback_title = payload.job_title or "software engineer"
    skills = payload.skills or []
    skill_1 = _compact_query(str(skills[0]), fallback="python") if len(skills) > 0 else "python"
    skill_2 = _compact_query(str(skills[1]), fallback="backend") if len(skills) > 1 else "backend"

    raw_queries = [
        f"{fallback_title}",
        f"{skill_1} {fallback_title}",
        f"{skill_2} {fallback_title} remote",
    ]
    q1, q2, q3 = _pick_three_queries(raw_queries, fallback_title)

    results = await asyncio.gather(
        search_linkedin_jobs(q1, location=payload.location, max_results=10),
        search_linkedin_jobs(q2, location=payload.location, max_results=10),
        search_linkedin_jobs(q3, location=payload.location, max_results=10),
        return_exceptions=True,
    )

    all_jobs: list[dict] = []
    for result in results:
        if isinstance(result, list):
            all_jobs.extend(result)

    if len(all_jobs) < 5:
        broad_query = _compact_query(fallback_title, fallback="software engineer")
        try:
            extra = await search_linkedin_jobs(
                broad_query,
                location=payload.location,
                max_results=15,
            )
            all_jobs.extend(extra)
        except Exception:
            pass

    jobs = _dedupe_jobs(all_jobs)
    scored = await asyncio.gather(
        *(score_job_against_cv(job, profile, payload.job_title) for job in jobs[:40])
    ) if jobs else []

    scored_filtered = [j for j in scored if j.get("relevance_score", 0) >= 25]
    if not scored_filtered and scored:
        scored_filtered = sorted(scored, key=lambda x: x.get("relevance_score", 0), reverse=True)[:3]

    scored_sorted = sorted(scored_filtered, key=lambda x: x.get("relevance_score", 0), reverse=True)
    selected = scored_sorted[: min(payload.max_results, len(scored_sorted))]

    if not selected:
        provider_reason = get_linkedin_provider_reason()
        message = f"No matching jobs found. Reason: {provider_reason}"
        await emit("job_search", "NO_JOBS_FOUND", {"message": message}, "error")
        await emit("orchestrator", "AGENT_ERROR", {"agent": "job_search", "error": message}, "error")
        await emit("orchestrator", "RUN_CRASHED", {"error": message}, "error")

        run.status = "failed"
        run.summary = {**(run.summary or {}), "error": message, "jobs_found": 0}
        await db.commit()
        return {"run_id": payload.run_id, "jobs": [], "status": "failed", "error": message}

    normalized_jobs: list[dict] = []
    seen_ids: set[str] = set()
    for idx, raw_job in enumerate(selected):
        job = dict(raw_job)
        candidate_id = str(job.get("job_id") or job.get("id") or job.get("link") or "").strip()
        if not candidate_id:
            candidate_id = f"match_{idx}_{uuid4().hex[:8]}"
        final_id = candidate_id
        suffix = 1
        while final_id in seen_ids:
            final_id = f"{candidate_id}_{suffix}"
            suffix += 1
        seen_ids.add(final_id)
        job["job_id"] = final_id
        job["id"] = final_id
        normalized_jobs.append(job)

    await db.execute(delete(Job).where(Job.run_id == payload.run_id))
    for idx, job in enumerate(normalized_jobs):
        raw_job_id = str(job.get("job_id") or idx)
        db_job_id = f"{payload.run_id}:{raw_job_id}:{idx}"
        db.add(
            Job(
                id=db_job_id,
                run_id=payload.run_id,
                title=str(job.get("title", "")),
                company=str(job.get("company", "")),
                location=str(job.get("location", "")),
                description=str(job.get("description", "")),
                apply_url=str(job.get("apply_url", "") or job.get("link", "")),
                source=str(job.get("source", "linkedin")),
                relevance_score=float(job.get("relevance_score", 0.0) or 0.0),
            )
        )

    summary = run.summary or {}
    summary["jobs_found"] = len(normalized_jobs)
    summary["emails_found"] = 0
    summary["cvs_tailored"] = 0
    summary["resumes_tailored"] = 0
    summary["intake_stage"] = "awaiting_job_selection"
    run.summary = summary
    run.status = "awaiting_job_selection"
    run.phase = "awaiting_job_selection"
    await db.commit()

    for job in normalized_jobs[:20]:
        await emit(
            "job_search",
            "JOB_FOUND",
            {
                "job_id": job.get("job_id", ""),
                "title": job.get("title", ""),
                "company": job.get("company", ""),
                "location": job.get("location", ""),
                "source": job.get("source", "linkedin"),
                "relevance_score": float(job.get("relevance_score", 0.0) or 0.0),
            },
            "running",
        )

    await emit(
        "job_search",
        "SEARCH_COMPLETE",
        {"count_selected": len(normalized_jobs), "mode": "cv_intake"},
        "done",
    )
    await emit("orchestrator", "AGENT_DONE", {"agent": "job_search"}, "done")
    await emit(
        "orchestrator",
        "AWAITING_JOB_SELECTION",
        {
            "run_id": payload.run_id,
            "jobs_found": len(normalized_jobs),
            "message": "Select one or more jobs to continue with email finder.",
        },
        "done",
    )

    return {"run_id": payload.run_id, "jobs": normalized_jobs, "status": "awaiting_job_selection"}

@router.post("/start-run")
async def start_run_with_selected_jobs(payload: CVStartRunRequest, db: AsyncSession = Depends(get_db)):
    """Continue an existing intake run (preferred) or create a new one if run_id is not provided."""
    selected_jobs = _normalize_selected_jobs(payload.selected_jobs)
    if not selected_jobs:
        raise HTTPException(status_code=400, detail="No valid selected jobs provided.")

    existing_run: Run | None = None
    if payload.run_id:
        existing_run = await db.get(Run, payload.run_id)
        if not existing_run:
            raise HTTPException(status_code=404, detail="Run not found.")

        if existing_run.status not in {"pending", "awaiting_job_selection", "failed"}:
            raise HTTPException(
                status_code=400,
                detail=f"Run {existing_run.id} is not ready for job selection (status={existing_run.status}).",
            )

        existing_task = RUN_TASKS.get(existing_run.id)
        if existing_task and not existing_task.done():
            raise HTTPException(
                status_code=409,
                detail=f"Run {existing_run.id} is already running.",
            )

    if existing_run is None:
        await _resolve_running_conflict(db)
        run_id = str(uuid4())
        run = Run(
            id=run_id,
            created_at=datetime.utcnow(),
            status="pending",
            phase="phase1",
            user_profile={},
            preferences={},
            summary={},
        )
        db.add(run)
        existing_run = run

    profile_data = payload.profile if isinstance(payload.profile, dict) else {}
    if not profile_data and isinstance(existing_run.user_profile, dict):
        profile_data = existing_run.user_profile
    await _save_profile(db, profile_data)

    existing_run.user_profile = profile_data
    existing_run.preferences = payload.preferences if isinstance(payload.preferences, dict) else {}
    run_summary = existing_run.summary or {}
    run_summary["dry_run"] = bool(payload.dry_run)
    run_summary["seed_jobs"] = len(selected_jobs)
    run_summary["selected_jobs"] = len(selected_jobs)
    run_summary["skip_cv_approval"] = True
    run_summary["skip_email_approval"] = True
    existing_run.summary = run_summary
    existing_run.status = "pending"
    existing_run.phase = "phase1"
    await db.commit()

    task = asyncio.create_task(_execute_run_with_jobs(existing_run.id, payload.dry_run, selected_jobs))
    RUN_TASKS[existing_run.id] = task

    return {"run_id": existing_run.id}

# --- Helper logic for skipping job search phase ---
from api.routes import _emit_step, _persist_run_outputs

async def _execute_run_with_jobs(run_id: str, dry_run: bool, selected_jobs: list[dict]) -> None:
    from agents.orchestrator import run_orchestrator_phase1_with_jobs

    try:
        logger.info("CV intake run starting: run_id=%s selected_jobs=%s", run_id, len(selected_jobs))

        async with async_session() as db:
            run = await db.get(Run, run_id)
            if not run:
                logger.error("CV intake run not found: %s", run_id)
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
            "resume_raw_text": str(profile.get("raw_text", "")),
            "jobs": selected_jobs,
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

        async def emit(agent: str, event_type: str, data: dict, status: str = "running") -> None:
            await _emit_step(run_id, agent, event_type, data, status)

        result = await run_orchestrator_phase1_with_jobs(initial_state, emit)
        await _persist_run_outputs(run_id, result)
        if bool((result or {}).get("skip_cv_approval", initial_state.get("skip_cv_approval", False))):
            from api.routes import _execute_phase2

            await _execute_phase2(run_id)
        logger.info("CV intake run completed: %s", run_id)
    except Exception as exc:
        logger.exception("CV intake run crashed: %s", run_id)
        try:
            await _emit_step(run_id, "orchestrator", "RUN_CRASHED", {"error": str(exc)}, "error")
        except Exception:
            logger.exception("Failed to emit crash step for run %s", run_id)

        async with async_session() as db:
            run = await db.get(Run, run_id)
            if run:
                run.status = "failed"
                summary = run.summary or {}
                summary["error"] = str(exc)
                run.summary = summary
                await db.commit()
