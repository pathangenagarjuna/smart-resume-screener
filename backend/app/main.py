from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List

from . import models, schemas
from .database import engine, get_db, Base
from .resume_parser import extract_resume_text
from .llm_service import analyze_resume

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Smart Resume Screener API",
    description="Parses resumes, matches them to a job description with an LLM, and ranks candidates.",
    version="1.0.0",
)

# Allow the local frontend (or any dev frontend) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://smart-resume-screener.netlify.app",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "service": "smart-resume-screener"}


# ---------- Jobs ----------

@app.post("/jobs", response_model=schemas.JobOut)
def create_job(job: schemas.JobCreate, db: Session = Depends(get_db)):
    db_job = models.Job(title=job.title, description=job.description)
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job


@app.get("/jobs", response_model=List[schemas.JobOut])
def list_jobs(db: Session = Depends(get_db)):
    return db.query(models.Job).order_by(desc(models.Job.created_at)).all()


@app.get("/jobs/{job_id}", response_model=schemas.JobOut)
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# ---------- Resumes ----------

@app.post("/jobs/{job_id}/resumes", response_model=schemas.ResumeOut)
async def upload_resume(job_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Uploads a single resume (PDF/DOCX/TXT) for a job, extracts raw text,
    then calls the LLM to extract structured data AND compute the match
    score + justification in one step.
    """
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    file_bytes = await file.read()
    try:
        raw_text = extract_resume_text(file.filename, file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    db_resume = models.Resume(
        job_id=job_id,
        filename=file.filename,
        raw_text=raw_text,
        status="uploaded",
    )
    db.add(db_resume)
    db.commit()
    db.refresh(db_resume)

    # Run LLM analysis immediately (extraction + scoring in one call)
    try:
        analysis = analyze_resume(raw_text, job.description)
        db_resume.candidate_name = analysis["candidate_name"]
        db_resume.skills = analysis["skills"]
        db_resume.experience = analysis["experience"]
        db_resume.education = analysis["education"]
        db_resume.match_score = analysis["match_score"]
        db_resume.justification = analysis["justification"]
        db_resume.status = "scored"
        db.commit()
        db.refresh(db_resume)
    except Exception as e:
        db_resume.status = "parse_failed"
        db.commit()
        raise HTTPException(status_code=502, detail=f"LLM analysis failed: {str(e)}")

    return db_resume


@app.get("/jobs/{job_id}/candidates", response_model=List[schemas.ResumeListItem])
def list_candidates(job_id: int, db: Session = Depends(get_db)):
    """
    Returns all resumes for a job, ranked highest match_score first.
    This is the "shortlisted candidates with justification" view.
    """
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    resumes = (
        db.query(models.Resume)
        .filter(models.Resume.job_id == job_id)
        .order_by(desc(models.Resume.match_score))
        .all()
    )
    return resumes


@app.get("/resumes/{resume_id}", response_model=schemas.ResumeOut)
def get_resume(resume_id: int, db: Session = Depends(get_db)):
    resume = db.query(models.Resume).filter(models.Resume.id == resume_id).first()
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    return resume


@app.delete("/resumes/{resume_id}")
def delete_resume(resume_id: int, db: Session = Depends(get_db)):
    resume = db.query(models.Resume).filter(models.Resume.id == resume_id).first()
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    db.delete(resume)
    db.commit()
    return {"status": "deleted"}
