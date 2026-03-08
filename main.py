from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as routes_router
from api.cv_intake import router as cv_intake_router
from models.database import create_tables

app = FastAPI(title="Job Hunter Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_router, prefix="/api")
app.include_router(cv_intake_router, prefix="/api/cv-intake")


@app.on_event("startup")
async def startup() -> None:
    await create_tables()
    import os

    os.makedirs("uploads", exist_ok=True)
    os.makedirs("uploads/resumes", exist_ok=True)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}