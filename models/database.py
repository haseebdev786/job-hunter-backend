from __future__ import annotations

from datetime import datetime
from typing import AsyncGenerator
from uuid import uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config import settings


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    phase: Mapped[str] = mapped_column(String(32), default="phase1", nullable=False)
    user_profile: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    preferences: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    summary: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "status": self.status,
            "phase": self.phase,
            "user_profile": self.user_profile,
            "preferences": self.preferences,
            "summary": self.summary,
        }


class Step(Base):
    __tablename__ = "steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("runs.id"), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    agent: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    data: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="running", nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "agent": self.agent,
            "event_type": self.event_type,
            "data": self.data,
            "status": self.status,
        }


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("runs.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    company: Mapped[str] = mapped_column(String(256), nullable=False)
    location: Mapped[str] = mapped_column(String(256), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    apply_url: Mapped[str] = mapped_column(String(1024), default="")
    source: Mapped[str] = mapped_column(String(64), default="")
    relevance_score: Mapped[float] = mapped_column(Float, default=0.0)


class HREmail(Base):
    __tablename__ = "hr_emails"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("runs.id"), nullable=False, index=True)
    job_id: Mapped[str] = mapped_column(String(128), ForeignKey("jobs.id"), nullable=False, index=True)
    company: Mapped[str] = mapped_column(String(256), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String(64), default="")


class TailoredResume(Base):
    __tablename__ = "tailored_resumes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("runs.id"), nullable=False, index=True)
    job_id: Mapped[str] = mapped_column(String(128), ForeignKey("jobs.id"), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, default="")
    pdf_path: Mapped[str] = mapped_column(String(1024), default="")
    match_score: Mapped[float] = mapped_column(Float, default=0.0)


class SentEmail(Base):
    __tablename__ = "sent_emails"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("runs.id"), nullable=False, index=True)
    job_id: Mapped[str] = mapped_column(String(128), ForeignKey("jobs.id"), nullable=False, index=True)
    recipient: Mapped[str] = mapped_column(String(320), nullable=False)
    subject: Mapped[str] = mapped_column(String(512), default="")
    body: Mapped[str] = mapped_column(Text, default="")
    sent_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="sent")


class CVApproval(Base):
    __tablename__ = "cv_approvals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("runs.id"), nullable=False, index=True)
    job_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    original_content: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    edited_content: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    company: Mapped[str] = mapped_column(String(256), default="")
    job_title: Mapped[str] = mapped_column(String(256), default="")
    match_score: Mapped[float] = mapped_column(Float, default=0.0)
    pdf_path: Mapped[str] = mapped_column(String(1024), default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self) -> dict:
        content = self.edited_content if self.edited_content else self.original_content
        return {
            "id": self.id,
            "run_id": self.run_id,
            "job_id": self.job_id,
            "status": self.status,
            "content": content,
            "original_content": self.original_content,
            "edited_content": self.edited_content,
            "company": self.company,
            "job_title": self.job_title,
            "match_score": self.match_score,
            "pdf_path": self.pdf_path,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class EmailApproval(Base):
    __tablename__ = "email_approvals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("runs.id"), nullable=False, index=True)
    job_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    recipient: Mapped[str] = mapped_column(String(320), default="")
    original_subject: Mapped[str] = mapped_column(String(512), default="")
    original_body: Mapped[str] = mapped_column(Text, default="")
    edited_subject: Mapped[str] = mapped_column(String(512), default="")
    edited_body: Mapped[str] = mapped_column(Text, default="")
    company: Mapped[str] = mapped_column(String(256), default="")
    job_title: Mapped[str] = mapped_column(String(256), default="")
    attachment_path: Mapped[str] = mapped_column(String(1024), default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "job_id": self.job_id,
            "status": self.status,
            "recipient": self.recipient,
            "subject": self.edited_subject or self.original_subject,
            "body": self.edited_body or self.original_body,
            "original_subject": self.original_subject,
            "original_body": self.original_body,
            "edited_subject": self.edited_subject,
            "edited_body": self.edited_body,
            "company": self.company,
            "job_title": self.job_title,
            "attachment_path": self.attachment_path,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Profile(Base):
    __tablename__ = "profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    data: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)


engine = create_async_engine(settings.database_url, future=True)
async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def create_tables() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session
