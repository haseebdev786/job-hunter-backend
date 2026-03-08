from __future__ import annotations

from datetime import datetime
from typing import Any, Awaitable, Callable, List, NotRequired, TypedDict

from langgraph.graph import END, StateGraph

from agents.email_finder_agent import run_email_finder_agent
from agents.email_sender_agent import run_email_sender_agent, run_email_drafter_agent
from agents.job_search_agent import run_job_search_agent
from agents.resume_tailor_agent import run_resume_tailor_agent


class AgentState(TypedDict):
    run_id: str
    user_profile: dict
    preferences: dict
    resume_raw_text: str
    jobs: List[dict]
    hr_emails: List[dict]
    tailored_resumes: dict
    sent_emails: List[dict]
    draft_emails: List[dict]
    steps: List[dict]
    current_agent: str
    dry_run: bool
    phase: str
    errors: List[str]
    skip_cv_approval: NotRequired[bool]
    skip_email_approval: NotRequired[bool]
    approved_emails: NotRequired[List[dict]]


EmitCallable = Callable[[str, str, dict[str, Any], str], Awaitable[None]]

import logging
logger = logging.getLogger(__name__)


async def _guarded_run(
    agent_name: str,
    fn: Callable[[dict[str, Any], EmitCallable], Awaitable[dict[str, Any]]],
    state: dict[str, Any],
    emit: EmitCallable,
) -> dict[str, Any]:
    try:
        logger.debug("Starting agent: %s", agent_name)
        state["current_agent"] = agent_name
        await emit("orchestrator", "AGENT_START", {"agent": agent_name}, "running")
        updated = await fn(state, emit)
        await emit("orchestrator", "AGENT_DONE", {"agent": agent_name}, "done")
        logger.debug("Agent %s completed successfully.", agent_name)
        return updated
    except Exception as exc:
        logger.error("Agent %s failed: %s", agent_name, exc)
        state.setdefault("errors", []).append(f"{agent_name}: {exc}")
        await emit(
            "orchestrator",
            "AGENT_ERROR",
            {"agent": agent_name, "error": str(exc)},
            "error",
        )
        return state


# ─── Phase 1: Job Search → Email Finder → Resume Tailor → PAUSE ───
def build_phase1_graph(emit: EmitCallable):
    graph = StateGraph(AgentState)

    async def run_job_search_node(state: AgentState) -> AgentState:
        return await _guarded_run("job_search", run_job_search_agent, state, emit)

    async def run_email_finder_node(state: AgentState) -> AgentState:
        return await _guarded_run("email_finder", run_email_finder_agent, state, emit)

    async def run_resume_tailor_node(state: AgentState) -> AgentState:
        return await _guarded_run("resume_tailor", run_resume_tailor_agent, state, emit)

    graph.add_node("job_search", run_job_search_node)
    graph.add_node("email_finder", run_email_finder_node)
    graph.add_node("resume_tailor", run_resume_tailor_node)

    graph.set_entry_point("job_search")
    graph.add_edge("job_search", "email_finder")
    graph.add_edge("email_finder", "resume_tailor")
    graph.add_edge("resume_tailor", END)

    return graph.compile()


# ─── Phase 1 (Variation): Jobs Pre-provided → Email Finder → Resume Tailor → PAUSE ───
def build_phase1_with_jobs_graph(emit: EmitCallable):
    graph = StateGraph(AgentState)

    async def run_email_finder_node(state: AgentState) -> AgentState:
        return await _guarded_run("email_finder", run_email_finder_agent, state, emit)

    async def run_resume_tailor_node(state: AgentState) -> AgentState:
        return await _guarded_run("resume_tailor", run_resume_tailor_agent, state, emit)

    graph.add_node("email_finder", run_email_finder_node)
    graph.add_node("resume_tailor", run_resume_tailor_node)

    graph.set_entry_point("email_finder")
    graph.add_edge("email_finder", "resume_tailor")
    graph.add_edge("resume_tailor", END)

    return graph.compile()


# ─── Phase 2: Email Drafter (draft only, no send) → PAUSE ───
def build_phase2_graph(emit: EmitCallable):
    graph = StateGraph(AgentState)

    async def run_email_drafter_node(state: AgentState) -> AgentState:
        return await _guarded_run("email_sender", run_email_drafter_agent, state, emit)

    graph.add_node("email_drafter", run_email_drafter_node)
    graph.set_entry_point("email_drafter")
    graph.add_edge("email_drafter", END)

    return graph.compile()


# ─── Phase 3: Email Sender (send approved emails only) ───
def build_phase3_graph(emit: EmitCallable):
    graph = StateGraph(AgentState)

    async def run_email_sender_node(state: AgentState) -> AgentState:
        return await _guarded_run("email_sender", run_email_sender_agent, state, emit)

    graph.add_node("email_sender", run_email_sender_node)
    graph.set_entry_point("email_sender")
    graph.add_edge("email_sender", END)

    return graph.compile()


async def run_orchestrator_phase1(state: AgentState, emit: EmitCallable) -> AgentState:
    """Phase 1: Search jobs, find emails, tailor resumes, then pause for CV approval."""
    await emit(
        "orchestrator",
        "RUN_STARTED",
        {"run_id": state.get("run_id"), "timestamp": datetime.utcnow().isoformat(), "phase": "phase1"},
        "running",
    )
    state["phase"] = "phase1"
    app = build_phase1_graph(emit)
    result = await app.ainvoke(state)

    skip_cv_approval = bool(state.get("skip_cv_approval", False))
    if skip_cv_approval:
        await emit(
            "orchestrator",
            "CV_APPROVAL_SKIPPED",
            {
                "run_id": state.get("run_id"),
                "jobs_found": len(result.get("jobs", [])),
                "cvs_tailored": len(result.get("tailored_resumes", {})),
                "message": "CV approval skipped. Moving to email drafting.",
            },
            "done",
        )
        result["phase"] = "phase2"
    else:
        # Signal pause for CV approval
        await emit(
            "orchestrator",
            "AWAITING_CV_APPROVAL",
            {
                "run_id": state.get("run_id"),
                "jobs_found": len(result.get("jobs", [])),
                "cvs_tailored": len(result.get("tailored_resumes", {})),
                "message": "CVs are ready for your review. Please approve or edit each CV before proceeding.",
            },
            "done",
        )
        result["phase"] = "awaiting_cv_approval"
    return result


async def run_orchestrator_phase1_with_jobs(state: AgentState, emit: EmitCallable) -> AgentState:
    """Phase 1 Variation: Skip Job Search, start from Email Finder."""
    await emit(
        "orchestrator",
        "RUN_STARTED",
        {"run_id": state.get("run_id"), "timestamp": datetime.utcnow().isoformat(), "phase": "phase1"},
        "running",
    )
    state["phase"] = "phase1"
    app = build_phase1_with_jobs_graph(emit)
    result = await app.ainvoke(state)

    skip_cv_approval = bool(state.get("skip_cv_approval", False))
    if skip_cv_approval:
        await emit(
            "orchestrator",
            "CV_APPROVAL_SKIPPED",
            {
                "run_id": state.get("run_id"),
                "jobs_found": len(result.get("jobs", [])),
                "cvs_tailored": len(result.get("tailored_resumes", {})),
                "message": "CV approval skipped. Moving to email drafting.",
            },
            "done",
        )
        result["phase"] = "phase2"
    else:
        # Signal pause for CV approval
        await emit(
            "orchestrator",
            "AWAITING_CV_APPROVAL",
            {
                "run_id": state.get("run_id"),
                "jobs_found": len(result.get("jobs", [])),
                "cvs_tailored": len(result.get("tailored_resumes", {})),
                "message": "CVs are ready for your review. Please approve or edit each CV before proceeding.",
            },
            "done",
        )
        result["phase"] = "awaiting_cv_approval"
    return result


async def run_orchestrator_phase2(state: AgentState, emit: EmitCallable) -> AgentState:
    """Phase 2: Draft emails for approved CVs, then pause for email approval."""
    await emit(
        "orchestrator",
        "PHASE2_STARTED",
        {"run_id": state.get("run_id"), "phase": "phase2"},
        "running",
    )
    state["phase"] = "phase2"
    app = build_phase2_graph(emit)
    result = await app.ainvoke(state)

    skip_email_approval = bool(state.get("skip_email_approval", False))
    if skip_email_approval:
        await emit(
            "orchestrator",
            "EMAIL_APPROVAL_SKIPPED",
            {
                "run_id": state.get("run_id"),
                "emails_drafted": len(result.get("draft_emails", [])),
                "message": "Email approval skipped. Moving to sending phase.",
            },
            "done",
        )
        result["phase"] = "phase3"
    else:
        await emit(
            "orchestrator",
            "AWAITING_EMAIL_APPROVAL",
            {
                "run_id": state.get("run_id"),
                "emails_drafted": len(result.get("draft_emails", [])),
                "message": "Email drafts are ready. Please review and approve each email before sending.",
            },
            "done",
        )
        result["phase"] = "awaiting_email_approval"
    return result


async def run_orchestrator_phase3(state: AgentState, emit: EmitCallable) -> AgentState:
    """Phase 3: Send only approved emails."""
    await emit(
        "orchestrator",
        "PHASE3_STARTED",
        {"run_id": state.get("run_id"), "phase": "phase3"},
        "running",
    )
    state["phase"] = "phase3"
    app = build_phase3_graph(emit)
    result = await app.ainvoke(state)

    await emit(
        "orchestrator",
        "RUN_FINISHED",
        {
            "run_id": state.get("run_id"),
            "jobs": len(result.get("jobs", [])),
            "emails": len(result.get("sent_emails", [])),
        },
        "done",
    )
    result["phase"] = "done"
    return result


# Keep backward-compatible function for legacy code
async def run_orchestrator(state: AgentState, emit: EmitCallable) -> AgentState:
    """Run Phase 1 only (pipeline pauses after CV tailoring)."""
    return await run_orchestrator_phase1(state, emit)
