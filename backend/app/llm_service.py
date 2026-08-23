"""
LLM integration layer.

This module uses OpenRouter to:
1. Extract structured information from a resume.
2. Compare the resume with a job description.
3. Generate a match score and justification.

OpenRouter provides an OpenAI-compatible API.
"""

import os
import json
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# ---------------------------------------------------------
# OpenRouter configuration
# ---------------------------------------------------------

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

if API_KEY:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=API_KEY,
    )
else:
    client = None


# ---------------------------------------------------------
# System Prompt
# ---------------------------------------------------------

SYSTEM_PROMPT = """You are an expert technical recruiter and resume screening assistant.

You will be given:
1. A candidate's resume text
2. A job description

Your tasks are:

1. Extract structured information from the resume:
   - candidate_name
   - skills
   - experience
   - education

2. Compare the resume against the job description.

3. Give a match score from 1 to 10:
   - 1 = very poor fit
   - 5 = average fit
   - 10 = excellent fit

4. Give a concise justification of 2-4 sentences.
   Mention specific matching skills, relevant experience, and important
   missing requirements when possible.

IMPORTANT:
- Do not invent information that is not present in the resume.
- Only use evidence from the resume.
- Keep skills as a flat list of strings.
- Keep experience as a list of short strings.
- Keep education as a list of short strings.
- match_score must be an integer from 1 to 10.

Return ONLY one valid JSON object.

The JSON must have exactly this structure:

{
  "candidate_name": "string or null",
  "skills": ["skill1", "skill2"],
  "experience": ["experience entry 1", "experience entry 2"],
  "education": ["education entry 1", "education entry 2"],
  "match_score": 8,
  "justification": "Short explanation of why the candidate matches the job."
}
"""


# ---------------------------------------------------------
# User Prompt
# ---------------------------------------------------------

USER_PROMPT_TEMPLATE = """JOB DESCRIPTION:

{job_description}

---

RESUME TEXT:

{resume_text}

---

Analyze the resume against the job description.

Extract the candidate information, compare the candidate with the
job requirements, calculate a match score from 1 to 10, and provide
a short justification.

Return ONLY the JSON object.
"""


# ---------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------

def _extract_json(raw_text: str) -> dict:
    """
    Extract JSON from the model response.

    The model should return only JSON, but this function also handles
    cases where the model accidentally adds markdown code fences.
    """

    cleaned = raw_text.strip()

    # Remove markdown code fences
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Find JSON object if there is extra text
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1:
        cleaned = cleaned[start:end + 1]

    return json.loads(cleaned)


# ---------------------------------------------------------
# Resume analysis
# ---------------------------------------------------------

def analyze_resume(resume_text: str, job_description: str) -> dict:
    """
    Analyze a resume using OpenRouter.

    Returns:
        candidate_name
        skills
        experience
        education
        match_score
        justification
    """

    if not API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Please add OPENROUTER_API_KEY to backend/.env"
        )

    if not resume_text.strip():
        raise ValueError("Resume text is empty.")

    if not job_description.strip():
        raise ValueError("Job description is empty.")

    prompt = USER_PROMPT_TEMPLATE.format(
        job_description=job_description.strip(),
        resume_text=resume_text.strip()[:2000],
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=1500,
            temperature=0.2,
        )

    except Exception as e:
        raise RuntimeError(
            f"OpenRouter API request failed: {str(e)}"
        ) from e

    # Get model response
    raw_text = response.choices[0].message.content

    if not raw_text:
        raise ValueError("OpenRouter returned an empty response.")

    # Convert response into JSON
    try:
        result = _extract_json(raw_text)

    except (json.JSONDecodeError, AttributeError) as e:
        raise ValueError(
            "LLM did not return valid JSON. "
            f"Raw response: {raw_text[:1000]}"
        ) from e

    # -----------------------------------------------------
    # Basic validation / defaults
    # -----------------------------------------------------

    result.setdefault("candidate_name", None)
    result.setdefault("skills", [])
    result.setdefault("experience", [])
    result.setdefault("education", [])
    result.setdefault("match_score", None)
    result.setdefault("justification", "")

    # Make sure list fields are actually lists
    if not isinstance(result["skills"], list):
        result["skills"] = []

    if not isinstance(result["experience"], list):
        result["experience"] = []

    if not isinstance(result["education"], list):
        result["education"] = []

    # Convert score to number
    if result["match_score"] is not None:
        try:
            result["match_score"] = float(result["match_score"])

            # Keep score within 1-10
            result["match_score"] = max(
                1,
                min(10, result["match_score"])
            )

        except (TypeError, ValueError):
            result["match_score"] = None

    return result