from __future__ import annotations

import asyncio
import json
import logging

from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse

from models.database import Run, Step, async_session

logger = logging.getLogger(__name__)
POLL_INTERVAL_SECONDS = 0.6
TERMINAL_STATUSES = {"done", "failed"}


async def get_steps_since(run_id: str, last_step_id: int) -> list[Step]:
    async with async_session() as db:
        result = await db.execute(
            select(Step)
            .where(Step.run_id == run_id, Step.id > last_step_id)
            .order_by(Step.id.asc())
        )
        return list(result.scalars().all())


async def get_run(run_id: str) -> Run | None:
    async with async_session() as db:
        return await db.get(Run, run_id)


async def stream_run_events(run_id: str):
    logger.debug("Starting SSE stream for run_id=%s", run_id)
    last_step_id = 0

    async def generator():
        nonlocal last_step_id
        try:
            while True:
                new_steps = await get_steps_since(run_id, last_step_id)
                for step in new_steps:
                    yield {"event": "step", "data": json.dumps(step.to_dict())}
                    last_step_id = step.id

                run = await get_run(run_id)
                if run is None:
                    logger.error("SSE run not found: %s", run_id)
                    yield {
                        "event": "complete",
                        "data": json.dumps(
                            {
                                "run_id": run_id,
                                "status": "failed",
                                "summary": {"error": "Run not found"},
                            }
                        ),
                    }
                    return

                run_status = str(run.status or "").lower()
                if run_status in TERMINAL_STATUSES:
                    yield {
                        "event": "complete",
                        "data": json.dumps(
                            {
                                "run_id": run_id,
                                "status": run.status,
                                "summary": run.summary or {},
                                "phase": run.phase,
                            }
                        ),
                    }
                    return

                await asyncio.sleep(POLL_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Error in SSE stream for run_id=%s", run_id)
            yield {"event": "error", "data": json.dumps({"run_id": run_id, "error": str(exc)})}

    return EventSourceResponse(generator(), ping=15)
