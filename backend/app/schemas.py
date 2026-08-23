from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class JobCreate(BaseModel):
    title: str
    description: str


class JobOut(BaseModel):
    id: int
    title: str
    description: str
    created_at: datetime

    class Config:
        from_attributes = True


class ResumeOut(BaseModel):
    id: int
    job_id: int
    filename: str
    candidate_name: Optional[str] = None
    skills: Optional[List[str]] = None
    experience: Optional[List[str]] = None
    education: Optional[List[str]] = None
    match_score: Optional[float] = None
    justification: Optional[str] = None
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class ResumeListItem(BaseModel):
    id: int
    filename: str
    candidate_name: Optional[str] = None
    match_score: Optional[float] = None
    justification: Optional[str] = None
    status: str

    class Config:
        from_attributes = True
