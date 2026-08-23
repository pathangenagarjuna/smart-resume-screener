"""
LLM integration for Smart Resume Screener.

Uses OpenRouter to:
1. Extract structured information from a resume.
2. Compare the resume with a job description.
3. Generate a match score and justification.

The function always tries to return a clean Python dictionary.
"""

import os
import json
import re

from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openai/gpt-oss-20b:free"
)


# =========================================================
# OPENROUTER CLIENT
# =========================================================

if API_KEY:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=API_KEY,
        timeout=90.0,
    )
else:
    client = None


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are a resume screening system.

Your task is to compare a candidate resume with a job description.

IMPORTANT:
Return ONLY ONE COMPLETE VALID JSON OBJECT.

Do NOT return:
- Markdown
- ```json
- explanations before JSON
- explanations after JSON
- safety messages
- comments
- "User Safety"
- partial JSON

The response MUST contain exactly these fields:

{
  "candidate_name": "string or null",
  "skills": ["skill1", "skill2"],
  "experience": ["short experience 1", "short experience 2"],
  "education": ["short education 1"],
  "match_score": 8,
  "justification": "2-4 concise sentences explaining the match."
}

RULES:

1. candidate_name
Extract the candidate's name from the resume.
If it cannot be identified, use null.

2. skills
Return important technical and professional skills found in the resume.

3. experience
Return short descriptions of relevant internships, jobs, projects,
research work, or other experience found in the resume.

4. education
Return education information found in the resume.

5. match_score
Give an integer from 1 to 10.

6. justification
Write 2-4 concise sentences.
Explain:
- important matching skills
- relevant experience
- important missing requirements, if any

IMPORTANT:
- Never invent information.
- Use ONLY information present in the resume.
- Do not copy the entire resume.
- Keep arrays concise.
- Keep justification concise.
- The JSON must be COMPLETE.
- Close every JSON object and array before finishing.
"""


# =========================================================
# JSON EXTRACTION
# =========================================================

def _extract_json(raw_text: str) -> dict:

    if not raw_text:
        raise ValueError("Empty LLM response.")

    cleaned = raw_text.strip()

    print("\n========== RAW LLM RESPONSE ==========")
    print(cleaned)
    print("======================================\n")

    # Remove markdown code fences if model adds them
    cleaned = re.sub(
        r"```json\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"```\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = cleaned.strip()

    # Find JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "LLM did not return a complete JSON object. "
            f"Raw response: {cleaned[:1000]}"
        )

    json_text = cleaned[start:end + 1]

    try:
        result = json.loads(json_text)

    except json.JSONDecodeError as e:

        raise ValueError(
            "Invalid JSON returned by LLM. "
            f"JSON error: {e}. "
            f"Raw response: {cleaned[:1500]}"
        ) from e

    if not isinstance(result, dict):
        raise ValueError(
            "LLM response is not a JSON object."
        )

    return result


# =========================================================
# VALIDATE RESULT
# =========================================================

def _validate_result(result: dict) -> dict:

    if not isinstance(result, dict):
        raise ValueError(
            "LLM response is not a JSON object."
        )

    # -----------------------------------------------------
    # Default fields
    # -----------------------------------------------------

    result.setdefault("candidate_name", None)
    result.setdefault("skills", [])
    result.setdefault("experience", [])
    result.setdefault("education", [])
    result.setdefault("match_score", None)
    result.setdefault("justification", "")

    # -----------------------------------------------------
    # Candidate name
    # -----------------------------------------------------

    if result["candidate_name"] is not None:
        result["candidate_name"] = str(
            result["candidate_name"]
        ).strip()

    # -----------------------------------------------------
    # List fields
    # -----------------------------------------------------

    for field in [
        "skills",
        "experience",
        "education"
    ]:

        if not isinstance(result[field], list):
            result[field] = []

        result[field] = [
            str(item).strip()
            for item in result[field]
            if item is not None
        ]

    # -----------------------------------------------------
    # Match score
    # -----------------------------------------------------

    score = result.get("match_score")

    if score is not None:

        try:

            score = float(score)

            # Keep score between 1 and 10
            score = max(1, min(10, score))

            # Store integer if possible
            if score.is_integer():
                score = int(score)

            result["match_score"] = score

        except (TypeError, ValueError):

            result["match_score"] = None

    # -----------------------------------------------------
    # Justification
    # -----------------------------------------------------

    if result["justification"] is None:
        result["justification"] = ""

    result["justification"] = str(
        result["justification"]
    ).strip()

    return result


# =========================================================
# CALL LLM
# =========================================================

def _call_llm(prompt: str, max_tokens: int = 2500):

    if client is None:

        raise RuntimeError(
            "OpenRouter client was not initialized."
        )

    response = client.chat.completions.create(

        model=MODEL,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0,

        max_tokens=max_tokens,

        response_format={
            "type": "json_object"
        },

        extra_body={
            "reasoning": {
                "effort": "low"
            }
        }
    )

    if not response.choices:
        raise ValueError(
            "OpenRouter returned no choices."
        )

    message = response.choices[0].message

    raw_text = message.content

    if not raw_text:
        raise ValueError(
            "OpenRouter returned an empty response."
        )

    return raw_text


# =========================================================
# MAIN ANALYSIS FUNCTION
# =========================================================

def analyze_resume(
    resume_text: str,
    job_description: str
) -> dict:

    # -----------------------------------------------------
    # Check API key
    # -----------------------------------------------------

    if not API_KEY:

        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Please add OPENROUTER_API_KEY to backend/.env"
        )

    # -----------------------------------------------------
    # Validate inputs
    # -----------------------------------------------------

    if not resume_text or not resume_text.strip():

        raise ValueError(
            "Resume text is empty."
        )

    if not job_description or not job_description.strip():

        raise ValueError(
            "Job description is empty."
        )

    # -----------------------------------------------------
    # Limit input size
    # -----------------------------------------------------

    resume_text = resume_text.strip()[:8000]

    job_description = job_description.strip()[:5000]

    # -----------------------------------------------------
    # IMPORTANT:
    # Use an f-string here.
    # -----------------------------------------------------

    prompt = f"""
JOB DESCRIPTION:

{job_description}


RESUME:

{resume_text}


TASK:

Compare the resume with the job description.

Extract the candidate information and calculate a match score.

Return ONLY one complete valid JSON object.

Do not include markdown.

Do not include explanations outside JSON.

Make sure the JSON is completely closed before you finish.
"""

    print("\n====================================")
    print("STARTING LLM ANALYSIS")
    print("MODEL:", MODEL)
    print("====================================\n")

    # =====================================================
    # FIRST ATTEMPT
    # =====================================================

    try:

        raw_text = _call_llm(
            prompt,
            max_tokens=2500
        )

        print("\n========== FIRST MODEL OUTPUT ==========")
        print(raw_text)
        print("========================================\n")

        result = _extract_json(raw_text)

        result = _validate_result(result)

        print(
            "LLM ANALYSIS SUCCESS:",
            result
        )

        return result

    except Exception as first_error:

        print(
            "\nFIRST LLM ATTEMPT FAILED:",
            repr(first_error)
        )

    # =====================================================
    # RETRY
    # =====================================================

    retry_prompt = f"""
You are completing a resume screening task.

Return ONLY a COMPLETE VALID JSON object.

Required structure:

{{
  "candidate_name": "string or null",
  "skills": ["skill1", "skill2"],
  "experience": ["short experience"],
  "education": ["short education"],
  "match_score": 8,
  "justification": "2-4 concise sentences"
}}

IMPORTANT:

- Return valid JSON only.
- No markdown.
- No explanations.
- Do not stop halfway.
- Close all arrays.
- Close the JSON object.
- Keep the answer concise.
- Do not invent information.

JOB:

{job_description}

RESUME:

{resume_text}
"""

    print("\n====================================")
    print("RETRYING LLM ANALYSIS")
    print("====================================\n")

    try:

        raw_text = _call_llm(
            retry_prompt,
            max_tokens=3000
        )

        print("\n========== RETRY MODEL OUTPUT ==========")
        print(raw_text)
        print("========================================\n")

        result = _extract_json(raw_text)

        result = _validate_result(result)

        print(
            "LLM RETRY SUCCESS:",
            result
        )

        return result

    except Exception as retry_error:

        print(
            "\nLLM RETRY FAILED:",
            repr(retry_error)
        )

        raise RuntimeError(
            "LLM analysis failed after retry. "
            f"Reason: {retry_error}"
        ) from retry_error