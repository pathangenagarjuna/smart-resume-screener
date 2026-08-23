"""
LLM integration for Smart Resume Screener.

Uses OpenRouter (OpenAI-compatible API) to:
1. Extract structured information from a resume.
2. Compare the resume with a job description.
3. Generate a match score and justification.

The function always tries to return a clean Python dictionary.
"""

import os
import json
import re
import logging

from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError

load_dotenv()

# =========================================================
# LOGGING
# =========================================================
# Debug logging is OFF by default because resume text and job
# descriptions can contain PII. Turn it on locally by setting
# LLM_DEBUG=1 in your .env file.

logger = logging.getLogger("llm_service")
logging.basicConfig(level=logging.INFO)
DEBUG = os.getenv("LLM_DEBUG", "0") == "1"


def _debug(*parts: str) -> None:
    if DEBUG:
        logger.debug(" ".join(str(p) for p in parts))


# =========================================================
# ENVIRONMENT / CLIENT
# =========================================================

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")

client = (
    OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY, timeout=90.0)
    if API_KEY
    else None
)


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """You are a resume screening system.

Your task is to compare a candidate resume with a job description.

IMPORTANT:
Return ONLY ONE COMPLETE VALID JSON OBJECT. Nothing before it, nothing
after it - no markdown fences, no "here is the analysis", no reasoning
or thinking text, no explanations, no comments.

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
1. candidate_name - extract from the resume; use null if not identifiable.
2. skills - important technical and professional skills found in the resume.
3. experience - short descriptions of relevant jobs, internships, projects,
   or research found in the resume.
4. education - education info found in the resume.
5. match_score - an integer from 1 to 10.
6. justification - 2-4 concise sentences covering matching skills, relevant
   experience, and important missing requirements if any.

Never invent information. Use only what's in the resume. Keep arrays and
the justification concise. The JSON must be complete - close every object
and array before finishing."""


# =========================================================
# JSON EXTRACTION
# =========================================================

def _find_json_objects(text: str):
    """
    Scan text and yield every top-level {...} block found, by tracking
    brace depth and string state. More robust than find('{')/rfind('}'),
    which silently merges multiple blocks (e.g. a reasoning-model's
    "thinking" text plus its real answer) into one invalid span.
    """
    depth = 0
    start = None
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start:i + 1]
                    start = None


def _extract_json(raw_text: str) -> dict:
    """
    Extract the JSON object from a raw LLM response. Tries every
    top-level {...} block found and returns the last one that both
    parses as JSON and looks like our expected shape (has at least
    one of our known keys) - this handles models that prepend
    reasoning/thinking text before the actual answer.
    """
    if not raw_text:
        raise ValueError("Empty LLM response.")

    cleaned = raw_text.strip()
    cleaned = re.sub(r"```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()

    candidates = list(_find_json_objects(cleaned))
    if not candidates:
        raise ValueError(
            f"LLM did not return a JSON object. Raw response: {cleaned[:1000]}"
        )

    known_keys = {
        "candidate_name", "skills", "experience", "education",
        "match_score", "justification",
    }
    parsed_candidates = []
    for block in candidates:
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            parsed_candidates.append(parsed)

    if not parsed_candidates:
        raise ValueError(
            f"LLM did not return valid JSON. Raw response: {cleaned[:1500]}"
        )

    # Prefer the last block that actually looks like our schema (handles
    # "thinking" JSON fragments emitted before the real answer); fall back
    # to the last parseable block if none match the schema.
    for parsed in reversed(parsed_candidates):
        if known_keys & parsed.keys():
            return parsed
    return parsed_candidates[-1]


# =========================================================
# VALIDATE RESULT
# =========================================================

def _validate_result(result: dict) -> dict:
    if not isinstance(result, dict):
        raise ValueError("LLM response is not a JSON object.")

    result.setdefault("candidate_name", None)
    result.setdefault("skills", [])
    result.setdefault("experience", [])
    result.setdefault("education", [])
    result.setdefault("match_score", None)
    result.setdefault("justification", "")

    if result["candidate_name"] is not None:
        result["candidate_name"] = str(result["candidate_name"]).strip()

    for field in ["skills", "experience", "education"]:
        if not isinstance(result[field], list):
            result[field] = []
        result[field] = [str(item).strip() for item in result[field] if item is not None]

    score = result.get("match_score")
    if score is not None:
        try:
            score = float(score)
            score = max(1, min(10, score))
            if score.is_integer():
                score = int(score)
            result["match_score"] = score
        except (TypeError, ValueError):
            result["match_score"] = None

    if result["justification"] is None:
        result["justification"] = ""
    result["justification"] = str(result["justification"]).strip()

    return result


# =========================================================
# CALL LLM
# =========================================================

def _call_llm(prompt: str, max_tokens: int = 2500) -> str:
    if client is None:
        raise RuntimeError("OpenRouter client was not initialized.")

    # NOTE: response_format={"type": "json_object"} is intentionally not
    # used here - not every OpenRouter model (especially free-tier ones)
    # supports it, and an unsupported response_format fails the request
    # outright before the prompt-level JSON instructions even get a
    # chance to work. The system + user prompts already demand raw JSON.
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
    )

    if not response.choices:
        raise ValueError("OpenRouter returned no choices.")

    raw_text = response.choices[0].message.content
    if not raw_text:
        raise ValueError("OpenRouter returned an empty response.")

    return raw_text


def _build_prompt(job_description: str, resume_text: str, retry: bool = False) -> str:
    if not retry:
        return f"""JOB DESCRIPTION:

{job_description}

RESUME:

{resume_text}

TASK:
Compare the resume with the job description. Extract the candidate
information and calculate a match score. Return ONLY one complete valid
JSON object - no markdown, no explanations outside the JSON, no reasoning
text. Make sure the JSON is completely closed before you finish."""

    return f"""You are completing a resume screening task. Return ONLY a
COMPLETE VALID JSON object with exactly this structure:

{{
  "candidate_name": "string or null",
  "skills": ["skill1", "skill2"],
  "experience": ["short experience"],
  "education": ["short education"],
  "match_score": 8,
  "justification": "2-4 concise sentences"
}}

Return valid JSON only. No markdown. No explanations. No reasoning text.
Do not stop halfway - close all arrays and the JSON object. Do not invent
information; use only what's in the resume.

JOB:
{job_description}

RESUME:
{resume_text}"""


# =========================================================
# MAIN ANALYSIS FUNCTION
# =========================================================

def analyze_resume(resume_text: str, job_description: str) -> dict:
    if not API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Please add OPENROUTER_API_KEY "
            "to backend/.env (get one at https://openrouter.ai/keys)."
        )
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is empty.")
    if not job_description or not job_description.strip():
        raise ValueError("Job description is empty.")

    resume_text = resume_text.strip()[:8000]
    job_description = job_description.strip()[:5000]

    _debug("Starting LLM analysis with model:", MODEL)

    attempts = [
        _build_prompt(job_description, resume_text, retry=False),
        _build_prompt(job_description, resume_text, retry=True),
    ]

    last_error: Exception | None = None
    for attempt_num, prompt in enumerate(attempts, start=1):
        try:
            raw_text = _call_llm(prompt, max_tokens=2500 if attempt_num == 1 else 3000)
            _debug(f"Attempt {attempt_num} raw output:", raw_text)
            result = _extract_json(raw_text)
            result = _validate_result(result)
            _debug(f"Attempt {attempt_num} succeeded:", result)
            return result
        except AuthenticationError as e:
            # Bad/expired key can never succeed on retry - fail fast
            # instead of burning a second API call.
            raise RuntimeError(
                f"OpenRouter authentication failed - check OPENROUTER_API_KEY. ({e})"
            ) from e
        except Exception as e:
            _debug(f"Attempt {attempt_num} failed:", repr(e))
            last_error = e

    raise RuntimeError(f"LLM analysis failed after retry. Reason: {last_error}") from last_error