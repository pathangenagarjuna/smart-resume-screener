from sqlalchemy import Column, Integer, String, Text, Float, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
from datetime import datetime

from .database import Base


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    resumes = relationship("Resume", back_populates="job", cascade="all, delete-orphan")


class Resume(Base):
    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    candidate_name = Column(String(255), nullable=True)
    raw_text = Column(Text, nullable=False)

    # Structured data extracted by the LLM
    skills = Column(JSON, nullable=True)         # list[str]
    experience = Column(JSON, nullable=True)     # list[str] (roles / summaries)
    education = Column(JSON, nullable=True)      # list[str]

    # Matching results
    match_score = Column(Float, nullable=True)   # 1-10
    justification = Column(Text, nullable=True)

    status = Column(String(50), default="uploaded")  # uploaded -> parsed -> scored
    created_at = Column(DateTime, default=datetime.utcnow)

    job = relationship("Job", back_populates="resumes")
