"""
Groq Resume Parser — Llama 3.1 8B
Single-pass, full resume text, no token limits.
"""

import calendar
import json
import os
import re
import time
import requests


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Google AI Studio / Gemini API (serves open Gemma models, e.g. gemma-4-31b-it)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
GOOGLE_MODEL = os.environ.get("GOOGLE_MODEL", "gemma-4-31b-it")
GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Shared generation settings. Temperature/top_p are identical across providers
# so the bake-off is fair; only the output cap differs because Groq's free tier
# caps tokens-per-minute (TPM) at 6000 — reserving 12288 output tokens 413s
# before the request even runs. Google AI Studio (Gemma) has unlimited TPM.
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "12288"))
GROQ_MAX_OUTPUT_TOKENS = int(os.environ.get("GROQ_MAX_OUTPUT_TOKENS", "4096"))
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.1"))
TOP_P = float(os.environ.get("LLM_TOP_P", "0.9"))

# Request timeouts (seconds). Gemma 4 31B on the free tier is slow (~33 tok/s),
# so a full resume can take 2-4 min — give it a generous ceiling.
GROQ_TIMEOUT = int(os.environ.get("GROQ_TIMEOUT", "120"))
GOOGLE_TIMEOUT = int(os.environ.get("GOOGLE_TIMEOUT", "300"))

# Auto-retry transient provider failures (free-tier 500s, 503s, 429s, timeouts).
# This is what makes a live free-tier demo survive Google's intermittent errors.
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "4"))
LLM_RETRY_BACKOFF = float(os.environ.get("LLM_RETRY_BACKOFF", "5"))


SYSTEM_PROMPT = """You are an expert resume parser built for an Applicant Tracking System. You extract structured data from resumes with perfect accuracy. You must return ONLY valid JSON. No explanations, no markdown fences, no extra text."""

PARSE_PROMPT = """Extract ALL information from the resume below into the exact JSON structure shown.

RULES:
- Use null for missing fields, empty arrays [] where no items found.
- Be thorough — capture every skill, every experience, every detail.
- For RelevantJobTitles: list 3-5 synonymous job titles for the CurrentJobRole.
- ONLY extract skills that are explicitly written/named in the resume text. NEVER add skills not in the resume. Extract ONLY actual technologies, tools, and technical competencies — NOT section headers like "Technical Tools" or "Application Software", NOT generic terms like "learning" or "planning".
- Certifications and spoken languages are NOT skills — put them ONLY in their respective sections.
- For Languages: ONLY extract from an explicit "Languages" section listing spoken languages. Do NOT confuse programming languages with spoken languages. Do NOT add "English" just because the resume is in English. Return [] if no Languages section exists.
- For each experience, extract ONLY the job title written for that specific role — never copy titles across roles.
- For RelevantSkills per skill: list 1-3 related skills from the same technology family.
- For Education Type: infer "Full-time", "Part-time", "Online", or "Distance". Default "Full-time".
- For Certifications: split into name, issuing organization, and year.
- For OverallSummary.Summary: if the resume has an explicit Summary/Objective/Profile section, copy it ENTIRELY verbatim. If none exists, generate a 2-3 sentence summary.
- For Projects: extract ONLY projects that are EXPLICITLY named or listed in the resume (a dedicated Projects/Portfolio section, or a clearly named project/initiative within a role). NEVER invent a project by combining a company name with a generic phrase (e.g. "<Company> Digital Transformation", "<Company> Payment Platform"). NEVER turn a single word from a responsibility (e.g. "flexibility") into a project. If the resume has no explicitly named projects, return [].
- For Project descriptions: extract what the project does. Never leave Description empty if info exists.
- For Achievements: ONLY extract bullets with a specific number, percentage, or dollar amount. Max 5. Return [] if no quantified achievements. NEVER generate or reword.
- Extract ALL skills from skills/technologies sections and tables.
- Use JSON null (not string "null") for missing info.
- CRITICAL: Return ONLY valid JSON. No markdown, no explanations, no text before or after. Use straight quotes only. MAX 25 skills, MAX 5 responsibilities per role (1 short sentence each, max 15 words). MAX 3 RelevantSkills per skill. Keep KeyResponsibilities extremely concise.

{
  "PersonalDetails": {
    "FullName": "",
    "FirstName": "",
    "MiddleName": null,
    "LastName": "",
    "EmailID": "",
    "PhoneNumber": "",
    "CountryCode": "",
    "Location": "",
    "LinkedIn": "",
    "GitHub": "",
    "Portfolio": ""
  },
  "OverallSummary": {
    "Summary": "",
    "CurrentJobRole": "",
    "RelevantJobTitles": [],
    "TotalExperience": "",
    "Domain": ""
  },
  "ListOfExperiences": [
    {
      "JobTitle": "",
      "CompanyName": "",
      "Location": "",
      "StartDate": "",
      "EndDate": "",
      "EmploymentType": "",
      "ExperienceInYears": "",
      "Summary": "",
      "KeyResponsibilities": []
    }
  ],
  "ListOfSkills": [
    {
      "SkillName": "",
      "SkillExperienceInMonths": 0,
      "LastUsed": "",
      "RelevantSkills": []
    }
  ],
  "PrimarySkills": [],
  "SecondarySkills": [],
  "ListOfEducation": [
    {
      "Degree": "",
      "TypeOfEducation": "",
      "Field": "",
      "Institution": "",
      "Location": "",
      "YearPassed": "",
      "GPA": ""
    }
  ],
  "Certifications": [
    {
      "CertificationName": "",
      "IssuerName": "",
      "IssuedYear": ""
    }
  ],
  "Projects": [
    {
      "ProjectName": "",
      "Description": "",
      "CompanyWorked": "",
      "RoleInProject": "",
      "Technologies": [],
      "StartDate": "",
      "EndDate": "",
      "Link": ""
    }
  ],
  "Achievements": [],
  "Languages": []
}

RESUME:
---
RESUME_TEXT_HERE
---

Return ONLY the JSON object. No other text."""


def is_groq_configured():
    """Check if Groq API key is set."""
    return bool(GROQ_API_KEY)


def is_google_configured():
    """Check if Google AI Studio / Gemini API key is set."""
    return bool(GOOGLE_API_KEY)


_TRANSIENT_MARKERS = ("500", "503", "429", "Internal", "timed out", "timeout",
                      "temporarily", "overloaded", "unavailable", "request failed")


def _is_transient(err):
    """True if an error string looks like a retryable provider/capacity issue."""
    s = str(err)
    return any(m in s for m in _TRANSIENT_MARKERS)


def _infer_provider(model):
    """Guess the provider from a model name. Returns 'google', 'groq', or None."""
    if not model:
        return None
    m = model.lower()
    if "gemma" in m or "gemini" in m:
        return "google"
    if "llama" in m or "mixtral" in m or "qwen" in m:
        return "groq"
    return None


def _call_groq(system_prompt, user_prompt, model, key):
    """Call Groq's OpenAI-compatible chat endpoint.

    Returns a normalized dict: {content, usage, finish_reason} or {error}.
    """
    resp = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_completion_tokens": GROQ_MAX_OUTPUT_TOKENS,
        },
        timeout=GROQ_TIMEOUT,
    )
    if resp.status_code != 200:
        return {"error": f"Groq API error {resp.status_code}: {resp.text[:300]}"}

    data = resp.json()
    choice = data["choices"][0]
    usage = data.get("usage", {}) or {}
    return {
        "content": choice["message"]["content"],
        "finish_reason": choice.get("finish_reason", "unknown"),
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
    }


def _call_google(system_prompt, user_prompt, model, key):
    """Call Google AI Studio (Gemini API) — used for open Gemma models.

    Gemma models on the Gemini API do not accept a separate system role, so we
    fold the system prompt into the single user turn. Returns the same
    normalized dict as _call_groq.
    """
    url = GOOGLE_URL.format(model=model)
    combined = f"{system_prompt}\n\n{user_prompt}"
    resp = requests.post(
        url,
        headers={
            "x-goog-api-key": key,
            "Content-Type": "application/json",
        },
        json={
            "contents": [{"role": "user", "parts": [{"text": combined}]}],
            "generationConfig": {
                "temperature": TEMPERATURE,
                "topP": TOP_P,
                "maxOutputTokens": MAX_OUTPUT_TOKENS,
            },
        },
        timeout=GOOGLE_TIMEOUT,
    )
    if resp.status_code != 200:
        return {"error": f"Google API error {resp.status_code}: {resp.text[:300]}"}

    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        # Prompt blocked or empty — surface the feedback for debugging.
        feedback = data.get("promptFeedback", {})
        return {"error": f"Google API returned no candidates. Feedback: {json.dumps(feedback)[:200]}"}

    cand = candidates[0]
    parts = (cand.get("content") or {}).get("parts") or []
    content = "".join(p.get("text", "") for p in parts)
    usage = data.get("usageMetadata", {}) or {}
    return {
        "content": content,
        "finish_reason": cand.get("finishReason", "unknown"),
        "usage": {
            "prompt_tokens": usage.get("promptTokenCount"),
            "completion_tokens": usage.get("candidatesTokenCount"),
            "total_tokens": usage.get("totalTokenCount"),
        },
    }


def parse_resume(resume_text, model=None, api_key=None, provider=None):
    """
    Parse resume text using Groq (Llama) or Google AI Studio (Gemma).

    Args:
        resume_text: Full raw text from resume (no truncation)
        model: Override model name (e.g. "gemma-4-31b-it", "llama-3.1-8b-instant")
        api_key: Override API key
        provider: "groq" or "google". Inferred from model name if omitted.

    Returns:
        dict with parsed fields + _metadata
    """
    provider = (provider or _infer_provider(model)
                or os.environ.get("LLM_PROVIDER") or "groq").lower()

    if provider == "google":
        mdl = model or GOOGLE_MODEL
        key = api_key or GOOGLE_API_KEY
        if not key:
            return {"error": "GOOGLE_API_KEY not set. Set it as an env var or pass api_key."}
    else:
        mdl = model or GROQ_MODEL
        key = api_key or GROQ_API_KEY
        if not key:
            return {"error": "GROQ_API_KEY not set. Set it as an env var or pass api_key."}

    prompt = PARSE_PROMPT.replace("RESUME_TEXT_HERE", resume_text)
    start = time.time()

    try:
        # Retry transient provider failures (free-tier 500/503/429/timeout) with
        # linear backoff. A permanent error (e.g. bad key, 400) is returned at once.
        completion = None
        attempts = 0
        for attempt in range(LLM_MAX_RETRIES):
            attempts = attempt + 1
            try:
                if provider == "google":
                    completion = _call_google(SYSTEM_PROMPT, prompt, mdl, key)
                else:
                    completion = _call_groq(SYSTEM_PROMPT, prompt, mdl, key)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                completion = {"error": f"{provider} request failed: {e}"}

            if "error" not in completion:
                break
            if attempt < LLM_MAX_RETRIES - 1 and _is_transient(completion["error"]):
                time.sleep(LLM_RETRY_BACKOFF * (attempt + 1))
                continue
            break

        elapsed_ms = int((time.time() - start) * 1000)

        if "error" in completion:
            return {"error": completion["error"], "processing_time_ms": elapsed_ms,
                    "attempts": attempts}

        content = completion["content"]
        usage = completion["usage"]
        finish = completion["finish_reason"]

        parsed = _extract_json(content)

        if parsed is None:
            return {
                "error": "Failed to parse JSON from model response",
                "raw_response": content[:500],
                "finish_reason": finish,
                "processing_time_ms": elapsed_ms,
            }

        parsed["_metadata"] = {
            "parser": provider,
            "model": mdl,
            "processing_time_ms": elapsed_ms,
            "finish_reason": finish,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

        parsed = _post_process(parsed, resume_text)

        # Remove hallucinated skills not found in resume text
        try:
            _fix_skill_hallucination(parsed, resume_text)
        except Exception:
            pass

        # Deduplicate skills and split into Primary/Secondary (after hallucination filter)
        try:
            _fix_skill_dedup_and_cap(parsed)
        except Exception:
            pass

        # Enrich with taxonomy normalization (canonical IDs, categories, etc.)
        try:
            from groq_taxonomy import enrich_resume
            parsed = enrich_resume(parsed)
        except ImportError:
            pass

        return parsed

    except requests.exceptions.Timeout:
        limit = GOOGLE_TIMEOUT if provider == "google" else GROQ_TIMEOUT
        return {"error": f"{provider} API timed out after {limit}s", "processing_time_ms": int((time.time() - start) * 1000)}
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to {provider} API. Check your internet connection."}
    except Exception as e:
        return {"error": str(e), "processing_time_ms": int((time.time() - start) * 1000)}


def _sanitize_json_text(text):
    """Sanitize Unicode characters that break JSON parsing.

    Handles smart quotes, em/en dashes, and other common Unicode
    that LLMs copy from resume text into JSON string values.
    """
    # First pass: simple replacements that don't need string-state tracking
    text = text.replace('\u2018', "'").replace('\u2019', "'")   # smart single quotes
    text = text.replace('\u2013', '-').replace('\u2014', '-')   # en/em dashes
    text = text.replace('\u2026', '...')                         # ellipsis

    # Handle smart double quotes and unescaped straight quotes inside JSON strings.
    # Smart quotes (\u201c, \u201d) and bare " inside string values break JSON.
    # We track JSON string state to escape any problematic quotes.
    result = []
    in_json_str = False
    i = 0
    while i < len(text):
        c = text[i]
        if in_json_str and c == '\\' and i + 1 < len(text):
            result.append(c)
            result.append(text[i + 1])
            i += 2
            continue
        if c in ('\u201c', '\u201d'):
            # Smart double quotes are always inside string values — escape them
            result.append('\\"')
        elif c == '"':
            # Heuristic: is this a JSON structural quote or a stray quote inside a value?
            # JSON structural quotes are preceded by: {, [, ,, :, whitespace, or start of text
            # and followed by: }, ], ,, :, whitespace, or end of text
            if in_json_str:
                # We're inside a string. Check if this quote is structural (closes the string)
                # or stray (part of the text content).
                # Look ahead: if followed by , : } ] or whitespace+key pattern, it's structural.
                rest = text[i + 1:i + 10].lstrip()
                if rest and rest[0] in (',', ':', '}', ']', '\n'):
                    in_json_str = False
                    result.append(c)
                elif not rest:
                    # End of text — structural
                    in_json_str = False
                    result.append(c)
                else:
                    # Stray quote inside string value — escape it
                    result.append('\\"')
            else:
                in_json_str = True
                result.append(c)
        elif ord(c) < 32 and c not in ('\n', '\r', '\t'):
            result.append(' ')
        else:
            result.append(c)
        i += 1

    return ''.join(result)


def _repair_truncated_json(text):
    """Attempt to repair truncated JSON from a model that hit its token limit.

    Strategy: find the JSON start, walk character by character tracking
    nesting depth properly (handling strings and escapes), then close
    any unclosed structures at the truncation point.
    """
    start = text.find('{')
    if start == -1:
        return None

    fragment = text[start:]

    # Trim the ragged end: find the last complete JSON token boundary.
    # Walk backward from the end to find a safe cut point: after a
    # comma, colon, closing bracket/brace, or end of a string value.
    # This removes partial keys, values, or strings at the truncation edge.
    last_safe = len(fragment)
    for j in range(len(fragment) - 1, max(len(fragment) - 500, 0), -1):
        c = fragment[j]
        if c in (',', ':', ']', '}', '\n'):
            last_safe = j + 1
            break
        if c == '"':
            # Check if this quote ends a complete string value
            last_safe = j + 1
            break

    fragment = fragment[:last_safe].rstrip().rstrip(',')

    # Track nesting to know what closers are needed
    depth_brace = 0
    depth_bracket = 0
    in_str = False
    esc = False
    for c in fragment:
        if esc:
            esc = False
            continue
        if c == '\\' and in_str:
            esc = True
            continue
        if c == '"' and not esc:
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == '{':
            depth_brace += 1
        elif c == '}':
            depth_brace -= 1
        elif c == '[':
            depth_bracket += 1
        elif c == ']':
            depth_bracket -= 1

    # If we're inside an open string at EOF, close it
    if in_str:
        fragment += '"'

    # Close unclosed structures (innermost first)
    fragment += ']' * max(depth_bracket, 0)
    fragment += '}' * max(depth_brace, 0)

    try:
        return json.loads(fragment)
    except json.JSONDecodeError:
        pass

    # Last resort: try progressively stripping more from the end
    for trim in range(1, 50):
        lines = fragment.rsplit('\n', trim)
        if len(lines) > 1:
            candidate = lines[0].rstrip().rstrip(',')
            # Recount nesting
            db, dbk, ins, esc2 = 0, 0, False, False
            for c in candidate:
                if esc2:
                    esc2 = False
                    continue
                if c == '\\' and ins:
                    esc2 = True
                    continue
                if c == '"' and not esc2:
                    ins = not ins
                    continue
                if ins:
                    continue
                if c == '{':
                    db += 1
                elif c == '}':
                    db -= 1
                elif c == '[':
                    dbk += 1
                elif c == ']':
                    dbk -= 1
            if ins:
                candidate += '"'
            candidate += ']' * max(dbk, 0) + '}' * max(db, 0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    # Aggressive fallback for very long truncated responses:
    # Find the last complete "]," or "]" that closes a top-level array,
    # then close remaining sections with empty defaults.
    _SECTION_KEYS = [
        "PersonalDetails", "OverallSummary", "ListOfExperiences",
        "ListOfSkills", "PrimarySkills", "SecondarySkills",
        "ListOfEducation", "Certifications", "Projects",
        "Achievements", "Languages",
    ]

    # Strategy: find each top-level section's start position
    section_positions = []
    for key in _SECTION_KEYS:
        pattern = '"' + key + '"'
        idx = fragment.find(pattern)
        if idx >= 0:
            section_positions.append((idx, key))
    section_positions.sort()

    # Find the last section that was likely completed:
    # walk backwards through sections and try cutting after each one
    for i in range(len(section_positions) - 1, -1, -1):
        cut_key = section_positions[i][1]
        cut_idx = section_positions[i][0]

        # If there's a next section, cut just before it
        if i + 1 < len(section_positions):
            end_idx = section_positions[i + 1][0]
        else:
            end_idx = len(fragment)

        # Find the last "]" or "}" before the next section
        search_region = fragment[cut_idx:end_idx]
        last_close = max(search_region.rfind('],'), search_region.rfind('],\n'),
                         search_region.rfind(']\n'))
        if last_close < 0:
            last_close = max(search_region.rfind('},'), search_region.rfind('},\n'))
        if last_close < 0:
            continue

        candidate = fragment[:cut_idx + last_close + 1].rstrip().rstrip(',')

        # Add empty defaults for remaining sections
        found_keys = set()
        for pos, key in section_positions:
            if pos < cut_idx + last_close:
                found_keys.add(key)
        missing = []
        for key in _SECTION_KEYS:
            if key not in found_keys:
                if key in ("PersonalDetails", "OverallSummary"):
                    missing.append(f'  "{key}": {{}}')
                else:
                    missing.append(f'  "{key}": []')
        if missing:
            candidate += ',\n' + ',\n'.join(missing)
        candidate += '\n}'

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def _extract_json(text):
    """Robust JSON extraction from LLM response.

    Handles: clean JSON, markdown-wrapped JSON, smart quotes/Unicode,
    and truncated JSON from token-limited responses.
    """
    text = text.strip()

    # Step 0: Reasoning models (e.g. Gemma 4) emit prose/thinking BEFORE the
    # JSON. Try the outermost {...} span as raw JSON first — before whole-text
    # sanitizing, which mis-tracks quote state across the prose and corrupts
    # otherwise-valid JSON. Only returns on a successful parse, so it's safe.
    first_brace, last_brace = text.find('{'), text.rfind('}')
    if first_brace != -1 and last_brace > first_brace:
        span = text[first_brace:last_brace + 1]
        try:
            return json.loads(span)
        except json.JSONDecodeError:
            try:
                return json.loads(_sanitize_json_text(span))
            except json.JSONDecodeError:
                pass

    # Step 1: Sanitize Unicode characters
    text = _sanitize_json_text(text)

    # Step 2: Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Step 3: Try markdown code block extraction
    for pattern in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```']:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue

    # Step 4: Brace matching — find outermost complete { }
    start = text.find('{')
    if start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if esc:
                esc = False
                continue
            if c == '\\' and in_str:
                esc = True
                continue
            if c == '"' and not esc:
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

    # Step 5: Attempt to repair truncated JSON (from token limit cutoff)
    return _repair_truncated_json(text)


# --- Post-processing constants ---
MONTH_MAP = {}
for _i in range(1, 13):
    MONTH_MAP[calendar.month_name[_i].lower()] = _i
    MONTH_MAP[calendar.month_abbr[_i].lower()] = _i

_DEFAULT_EMPLOYMENT_TYPES = {"full-time", "full time", "fulltime"}


def _parse_date(date_str):
    """Convert LLM date strings to (year, month) tuple.

    Handles: "July 2021", "07/2021", "2021", "Present", "Jan 2020",
    "04/2015", "current", "till date", etc.
    Returns None on failure.
    """
    if not date_str or not isinstance(date_str, str):
        return None
    s = date_str.strip().lower()
    if s in ("present", "current", "till date", "now"):
        from datetime import date
        today = date.today()
        return (today.year, today.month)

    # Try "Month Year" — e.g. "July 2021", "Jan 2020"
    m = re.match(r'^([a-z]+)\s+(\d{4})$', s)
    if m:
        month_name, year_str = m.group(1), m.group(2)
        month = MONTH_MAP.get(month_name)
        if month:
            return (int(year_str), month)

    # Try "MM/YYYY" or "MM-YYYY"
    m = re.match(r'^(\d{1,2})[/\-](\d{4})$', s)
    if m:
        month, year = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return (year, month)

    # Try "YYYY-MM" or "YYYY/MM"
    m = re.match(r'^(\d{4})[/\-](\d{1,2})$', s)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return (year, month)

    # Try bare year "2021"
    m = re.match(r'^(\d{4})$', s)
    if m:
        return (int(m.group(1)), 1)

    return None


def _calc_months(start, end):
    """Return integer months between two (year, month) tuples. Minimum 1."""
    months = (end[0] - start[0]) * 12 + (end[1] - start[1])
    return max(months, 1)


def _fix_experience_years(parsed):
    """Recalculate ExperienceInYears for each experience from StartDate/EndDate."""
    experiences = parsed.get("ListOfExperiences")
    if not isinstance(experiences, list):
        return
    for exp in experiences:
        if not isinstance(exp, dict):
            continue
        start = _parse_date(exp.get("StartDate"))
        end = _parse_date(exp.get("EndDate"))
        if start and end:
            months = _calc_months(start, end)
            years = round(months / 12, 1)
            exp["ExperienceInYears"] = str(years)


def _fix_skill_experience(parsed):
    """Recalculate SkillExperienceInMonths by searching experience text for skill mentions."""
    skills = parsed.get("ListOfSkills")
    experiences = parsed.get("ListOfExperiences")
    if not isinstance(skills, list) or not isinstance(experiences, list):
        return

    # Pre-compute experience text blobs and durations
    exp_data = []
    for exp in experiences:
        if not isinstance(exp, dict):
            continue
        start = _parse_date(exp.get("StartDate"))
        end = _parse_date(exp.get("EndDate"))
        if not start or not end:
            continue
        months = _calc_months(start, end)
        # Combine job title + summary + responsibilities into searchable text
        parts = []
        title = exp.get("JobTitle")
        if isinstance(title, str):
            parts.append(title)
        summary = exp.get("Summary")
        if isinstance(summary, str):
            parts.append(summary)
        resps = exp.get("KeyResponsibilities")
        if isinstance(resps, list):
            for r in resps:
                if isinstance(r, str):
                    parts.append(r)
        text = " ".join(parts).lower()
        end_date_str = exp.get("EndDate", "")
        exp_data.append((text, months, end_date_str))

    for skill in skills:
        if not isinstance(skill, dict):
            continue
        name = skill.get("SkillName")
        if not isinstance(name, str) or len(name) < 2:
            continue
        # Build search names: main name + RelevantSkills
        search_names = [name.lower()]
        relevant = skill.get("RelevantSkills")
        if isinstance(relevant, list):
            for rs in relevant:
                if isinstance(rs, str) and len(rs) >= 2:
                    search_names.append(rs.lower())
        total_months = 0
        latest_end = None
        for text, months, end_date_str in exp_data:
            matched = any(sn in text for sn in search_names)
            if not matched:
                for sn in search_names:
                    if len(sn) <= 3:
                        try:
                            if re.search(r'\b' + re.escape(sn) + r'\b', text):
                                matched = True
                                break
                        except re.error:
                            pass
            if matched:
                total_months += months
                parsed_end = _parse_date(end_date_str)
                if parsed_end and (latest_end is None or parsed_end > latest_end):
                    latest_end = parsed_end
        if total_months > 0:
            skill["SkillExperienceInMonths"] = total_months
            if latest_end:
                skill["LastUsed"] = f"{calendar.month_name[latest_end[1]]} {latest_end[0]}"
        else:
            skill["SkillExperienceInMonths"] = None


def _fix_name_splitting(parsed):
    """Deterministically split FullName into First/Middle/Last by whitespace."""
    pd = parsed.get("PersonalDetails")
    if not isinstance(pd, dict):
        return
    full_name = pd.get("FullName")
    if not isinstance(full_name, str) or not full_name.strip():
        return
    parts = full_name.strip().split()
    if len(parts) == 1:
        pd["FirstName"] = parts[0].title()
        pd["MiddleName"] = None
        pd["LastName"] = parts[0].title()
    elif len(parts) == 2:
        pd["FirstName"] = parts[0].title()
        pd["MiddleName"] = None
        pd["LastName"] = parts[1].title()
    else:
        pd["FirstName"] = parts[0].title()
        pd["MiddleName"] = " ".join(parts[1:-1]).title()
        pd["LastName"] = parts[-1].title()
    pd["FullName"] = " ".join(p.title() for p in parts)



def _fix_phone_country_code(parsed):
    """Separate country code from phone number into its own field."""
    pd = parsed.get("PersonalDetails")
    if not isinstance(pd, dict):
        return
    phone = pd.get("PhoneNumber")
    if not isinstance(phone, str) or not phone.strip():
        return

    phone = phone.strip()

    # Extract leading country code: +1, +91, +44, +353, etc.
    m = re.match(r'^(\+\d{1,3})[\s.\-]+(.+)$', phone)
    if m:
        code = m.group(1)
        remaining = m.group(2).strip()
        pd["PhoneNumber"] = remaining
        # Only set CountryCode if not already populated
        existing_cc = pd.get("CountryCode")
        if not existing_cc or not str(existing_cc).strip() or str(existing_cc).strip().lower() in ("null", "none"):
            pd["CountryCode"] = code

    # Ensure CountryCode has + prefix if it's just digits
    cc = pd.get("CountryCode")
    if isinstance(cc, str) and cc.strip() and not cc.strip().startswith("+"):
        digits = re.sub(r'\D', '', cc)
        if digits and len(digits) <= 3:
            pd["CountryCode"] = f"+{digits}"


def _fix_merge_summary(parsed):
    """Merge Summary into KeyResponsibilities as first bullet, then clear it."""
    experiences = parsed.get("ListOfExperiences")
    if not isinstance(experiences, list):
        return
    for exp in experiences:
        if not isinstance(exp, dict):
            continue
        summary = exp.get("Summary")
        if not isinstance(summary, str) or not summary.strip():
            continue
        resps = exp.get("KeyResponsibilities")
        if not isinstance(resps, list):
            resps = []
        # Avoid duplicate if first bullet already matches summary
        if resps and isinstance(resps[0], str) and resps[0].strip() == summary.strip():
            pass  # already there as first bullet
        else:
            resps.insert(0, summary.strip())
        exp["KeyResponsibilities"] = resps


# Single-word "soft attribute" names that are never real projects.
_SOFT_ATTRIBUTE_NAMES = {
    "flexibility", "scalability", "reliability", "performance", "security",
    "usability", "maintainability", "availability", "efficiency", "quality",
    "productivity", "agility", "stability", "accuracy",
}
# Generic buzzwords — a project that is JUST "<Company> <these>" is fabricated.
_GENERIC_PROJECT_WORDS = {
    "digital", "transformation", "modernization", "modernisation", "platform",
    "platforms", "infrastructure", "solution", "solutions", "system", "systems",
    "initiative", "initiatives", "implementation", "application", "applications",
    "services", "enhancement", "enhancements",
}


def _fix_project_fabrication(parsed, resume_text=""):
    """Drop projects the model invented (not grounded in the resume).

    Two safe signals, tuned to never drop a real project:
      1. A single-word name that is a soft attribute ("Flexibility") lifted from
         a responsibility bullet.
      2. A "<CompanyName> <generic buzzwords>" template (e.g. "Frost Digital
         Platform Modernization") whose full name is NOT verbatim in the resume.
    """
    projects = parsed.get("Projects")
    if not isinstance(projects, list) or not resume_text:
        return
    text_lower = resume_text.lower()
    kept = []
    for proj in projects:
        if not isinstance(proj, dict):
            kept.append(proj)
            continue
        name = (proj.get("ProjectName") or "").strip()
        if not name:
            continue
        name_lower = name.lower()
        words = [w for w in re.split(r'[\s/,&\-]+', name_lower) if w]
        # Rule 1: single soft-attribute word
        if len(words) == 1 and words[0] in _SOFT_ATTRIBUTE_NAMES:
            continue
        # Verbatim in resume -> definitely real, keep
        if name_lower in text_lower:
            kept.append(proj)
            continue
        # Rule 2: starts with company name + remainder is all generic buzzwords
        company = (proj.get("CompanyWorked") or "").lower()
        co_words = set(company.split())
        remainder = [w for w in words if w not in co_words and len(w) > 3]
        starts_with_company = bool(words) and (words[0] in co_words or words[0] in company)
        if starts_with_company and remainder and all(w in _GENERIC_PROJECT_WORDS for w in remainder):
            continue
        kept.append(proj)
    parsed["Projects"] = kept


def _fix_project_company(parsed):
    """Infer CompanyWorked for projects from overlapping experience dates."""
    projects = parsed.get("Projects")
    experiences = parsed.get("ListOfExperiences")
    if not isinstance(projects, list) or not isinstance(experiences, list):
        return

    exp_ranges = []
    for exp in experiences:
        if not isinstance(exp, dict):
            continue
        start = _parse_date(exp.get("StartDate"))
        end = _parse_date(exp.get("EndDate"))
        company = exp.get("CompanyName", "")
        parts = []
        for r in (exp.get("KeyResponsibilities") or []):
            if isinstance(r, str):
                parts.append(r)
        if isinstance(exp.get("Summary"), str):
            parts.append(exp["Summary"])
        text = " ".join(parts).lower()
        exp_ranges.append((start, end, company, text))

    for proj in projects:
        if not isinstance(proj, dict):
            continue
        existing = proj.get("CompanyWorked")
        if isinstance(existing, str) and existing.strip() and existing.strip().lower() not in ("null", "none", "n/a"):
            continue

        proj_start = _parse_date(proj.get("StartDate"))
        proj_end = _parse_date(proj.get("EndDate"))
        best_company = None

        if proj_start and proj_end:
            for exp_start, exp_end, company, _ in exp_ranges:
                if exp_start and exp_end and company:
                    if proj_start <= exp_end and proj_end >= exp_start:
                        best_company = company
                        break

        if not best_company:
            techs = proj.get("Technologies") or []
            proj_name = (proj.get("ProjectName") or "").lower()
            for _, _, company, exp_text in exp_ranges:
                if not company:
                    continue
                matches = sum(1 for t in techs if isinstance(t, str) and t.lower() in exp_text)
                if matches >= 2 or (proj_name and len(proj_name) > 3 and proj_name in exp_text):
                    best_company = company
                    break

        if best_company:
            proj["CompanyWorked"] = best_company


_CERT_KEYWORDS = {"pmp", "scrum master", "csm", "ccna", "itil", "cissp", "cism", "cisa",
                   "aws certified", "azure certified", "safe", "six sigma", "prince2",
                   "togaf", "comptia", "certified scrum"}
_SOFT_SKILL_KEYWORDS = {"communication", "leadership", "teamwork", "problem solving",
                        "time management", "multi-tasking", "analytical skills",
                        "organizational skills", "detail-oriented", "interpersonal",
                        "adaptable", "creative", "organizational efficacy",
                        "process improvement", "process & organizational"}


_JOB_TITLE_KEYWORDS = {"manager", "engineer", "developer", "analyst", "architect",
                       "administrator", "consultant", "director", "lead", "specialist",
                       "coordinator", "supervisor", "officer", "executive", "intern",
                       "associate", "senior", "junior", "staff", "principal", "vp",
                       "president", "head of"}

# Section headers / category labels that LLMs extract as skills from resume tables
_SECTION_HEADER_KEYWORDS = {
    "technical tools", "application software", "operating systems",
    "programming languages", "other skills", "core competencies",
    "technical skills", "professional skills", "key skills",
    "areas of expertise", "tools & technologies", "tools and technologies",
    "software", "hardware", "databases", "frameworks", "methodologies",
    "platforms", "environments", "skill set", "competencies",
    "technologies used", "technical proficiency", "technical expertise",
    "additional skills", "relevant skills", "computer skills",
    "it skills", "languages and tools", "personal skills",
    "management skills", "soft skills", "professional summary",
    "tools", "technologies", "applications", "packages",
}

# Generic non-technical filler terms that are not real skills (exact match only).
# NOTE: keep this list to terms that are *never* a real tool. "Plan Plus" was
# removed — it is a legitimate PM tool (PlanPlus) and was being wrongly dropped
# when explicitly listed in a resume's skills section. The prompt + the
# grounding filter already keep junk out for capable models, so this denylist
# is only a light backstop.
_GENERIC_FILLER_KEYWORDS = {
    "learning", "lessons learned", "planning",
    "training", "best practices", "documentation", "reporting",
    "support", "maintenance", "operations", "administration",
    "collaboration", "optimization", "monitoring", "configuration",
    "implementation", "integration", "migration", "deployment",
}


def _fix_skill_contamination(parsed):
    """Remove certifications, soft skills, and job titles that leaked into ListOfSkills."""
    skills = parsed.get("ListOfSkills")
    if not isinstance(skills, list):
        return
    cleaned = []
    for skill in skills:
        if not isinstance(skill, dict):
            cleaned.append(skill)
            continue
        name = skill.get("SkillName", "")
        if not isinstance(name, str):
            cleaned.append(skill)
            continue
        name_lower = name.strip().lower()
        # Check if it's a certification
        if any(kw in name_lower for kw in _CERT_KEYWORDS):
            continue
        # Check if it's a soft skill
        if any(kw in name_lower for kw in _SOFT_SKILL_KEYWORDS):
            continue
        # Check if it's a job title (e.g., "Project Manager", "Software Developer")
        words = name_lower.split()
        if len(words) <= 3 and any(w in _JOB_TITLE_KEYWORDS for w in words):
            continue
        # Check if it's a section header / category label from resume tables
        if name_lower in _SECTION_HEADER_KEYWORDS:
            continue
        # Check if it's a generic non-technical filler term
        if name_lower in _GENERIC_FILLER_KEYWORDS:
            continue
        # Skip very long names (likely responsibility text, not a skill)
        if len(name) > 60:
            continue
        cleaned.append(skill)
    parsed["ListOfSkills"] = cleaned


def _fix_empty_certs(parsed):
    """Remove hallucinated placeholder certification objects with null/empty names."""
    certs = parsed.get("Certifications")
    if not isinstance(certs, list):
        return
    cleaned = []
    for cert in certs:
        if not isinstance(cert, dict):
            continue
        name = cert.get("CertificationName") or cert.get("Name") or ""
        if isinstance(name, str) and name.strip().lower() not in ("", "null", "none", "n/a"):
            cleaned.append(cert)
    parsed["Certifications"] = cleaned


def _find_skill_in_text(skill_name, text):
    """Check if a skill name appears in the resume text.

    Handles short names (C, R, Go) with word boundaries,
    special chars (C#, .NET, C++) with direct substring matching,
    and standard case-insensitive substring matching.
    """
    if not skill_name or not text:
        return False

    name = skill_name.strip()
    name_lower = name.lower()
    text_lower = text.lower()

    # Names with special chars (C#, C++, .NET, etc.) — use direct substring
    if any(c in name for c in '#.+/&'):
        return name_lower in text_lower

    # Very short names (1-2 chars like C, R) need word boundary matching
    if len(name) <= 2:
        try:
            return bool(re.search(r'\b' + re.escape(name_lower) + r'\b', text_lower))
        except re.error:
            return name_lower in text_lower

    # Direct case-insensitive substring match (handles most cases)
    if name_lower in text_lower:
        return True

    # Word-boundary match for medium-length names (avoids partial matches)
    if len(name) <= 20:
        try:
            if re.search(r'\b' + re.escape(name_lower) + r'\b', text_lower):
                return True
        except re.error:
            pass

    return False


def _fix_skill_hallucination(parsed, resume_text):
    """Remove skills whose SkillName does not appear in the resume text.

    The LLM sometimes generates plausible skills that aren't actually
    mentioned in the resume. This checks each SkillName against the
    original text and removes any that can't be found.
    """
    skills = parsed.get("ListOfSkills")
    if not isinstance(skills, list) or not resume_text:
        return

    cleaned = []
    for skill in skills:
        if not isinstance(skill, dict):
            cleaned.append(skill)
            continue
        name = skill.get("SkillName", "")
        if not isinstance(name, str) or not name.strip():
            continue  # drop skills with no name
        if _find_skill_in_text(name, resume_text):
            cleaned.append(skill)
        # else: skill not found in resume text, drop it

    parsed["ListOfSkills"] = cleaned


def _months_index(ym):
    """Convert a (year, month) tuple to an absolute month index for interval math."""
    return ym[0] * 12 + (ym[1] - 1)


def _fix_total_experience(parsed):
    """Calculate TotalExperience as the UNION of employment intervals.

    Summing the raw earliest-start -> latest-end span counts employment GAPS as
    worked time and over-states experience. Instead we merge overlapping date
    ranges and sum their lengths — this excludes gaps and avoids double-counting
    concurrent roles, matching how recruiters read "total experience".
    """
    experiences = parsed.get("ListOfExperiences")
    summary = parsed.get("OverallSummary")
    if not isinstance(experiences, list) or not isinstance(summary, dict):
        return

    intervals = []
    for exp in experiences:
        if not isinstance(exp, dict):
            continue
        start = _parse_date(exp.get("StartDate"))
        end = _parse_date(exp.get("EndDate"))
        if start and end:
            s, e = _months_index(start), _months_index(end)
            if e >= s:
                intervals.append((s, e))

    if not intervals:
        return

    # Merge overlapping/adjacent intervals, then sum their lengths.
    intervals.sort()
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    total_months = sum(e - s for s, e in merged)
    total_months = max(total_months, 1)
    years = round(total_months / 12, 1)
    summary["TotalExperience"] = f"{years} years"


def _fix_languages(parsed):
    """Normalize Languages array — convert objects to strings, remove empty."""
    langs = parsed.get("Languages")
    if not isinstance(langs, list):
        return
    cleaned = []
    for lang in langs:
        if isinstance(lang, str) and lang.strip():
            cleaned.append(lang.strip())
        elif isinstance(lang, dict):
            val = (lang.get("Language") or lang.get("language")
                   or lang.get("Name") or lang.get("name") or "")
            if isinstance(val, str) and val.strip():
                cleaned.append(val.strip())
    parsed["Languages"] = cleaned


_METRIC_PATTERN = re.compile(r'\d+[\d,]*\.?\d*\s*[%$]|\$[\d,]+|\d+[kKmM]\+?|\d{2,}')


def _fix_achievements(parsed):
    """Deduplicate achievements, validate metrics, cap at 5."""
    achievements = parsed.get("Achievements")
    if not isinstance(achievements, list):
        return
    seen = set()
    cleaned = []
    for ach in achievements:
        if isinstance(ach, str):
            desc = ach
        elif isinstance(ach, dict):
            desc = ach.get("Description") or ach.get("description") or ""
        else:
            continue
        if not isinstance(desc, str) or not desc.strip():
            continue
        key = desc.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        if _METRIC_PATTERN.search(desc):
            cleaned.append(desc.strip() if isinstance(ach, str) else ach)
    parsed["Achievements"] = cleaned[:5]


def _fix_location_cleanup(parsed):
    """Remove industry descriptors from location fields."""
    experiences = parsed.get("ListOfExperiences")
    if not isinstance(experiences, list):
        return
    for exp in experiences:
        if not isinstance(exp, dict):
            continue
        loc = exp.get("Location")
        if not isinstance(loc, str) or not loc.strip():
            continue
        cleaned = loc.strip()
        cleaned = re.sub(
            r'\s*\((?:Banking|Healthcare|Finance|Insurance|Entertainment|'
            r'Wireless|Retail|IT|Technology|Telecom|Manufacturing|'
            r'Energy|Automotive|Pharma|Media)[^)]*\)',
            '', cleaned, flags=re.IGNORECASE
        )
        company = exp.get("CompanyName", "")
        if isinstance(company, str) and company.strip() and len(company.strip()) > 2:
            cn = company.strip()
            if cn.lower() in cleaned.lower() and cn.lower() != cleaned.lower():
                cleaned = re.sub(re.escape(cn), '', cleaned, flags=re.IGNORECASE).strip()
                cleaned = re.sub(r'^[,\s\-/]+|[,\s\-/]+$', '', cleaned)
        if cleaned:
            exp["Location"] = cleaned

    # Clean education locations too
    education = parsed.get("ListOfEducation")
    if isinstance(education, list):
        for edu in education:
            if not isinstance(edu, dict):
                continue
            loc = edu.get("Location")
            inst = edu.get("Institution")
            if isinstance(loc, str) and isinstance(inst, str) and inst.strip():
                if inst.strip().lower() in loc.lower() and inst.strip().lower() != loc.lower():
                    cleaned = re.sub(re.escape(inst.strip()), '', loc, flags=re.IGNORECASE).strip()
                    cleaned = re.sub(r'^[,\s\-/]+|[,\s\-/]+$', '', cleaned)
                    if cleaned:
                        edu["Location"] = cleaned


def _fix_overall_responsibilities(parsed):
    """Aggregate top responsibilities from all roles into a top-level field."""
    experiences = parsed.get("ListOfExperiences")
    if not isinstance(experiences, list):
        return
    all_resps = []
    seen = set()
    for exp in experiences:
        if not isinstance(exp, dict):
            continue
        resps = exp.get("KeyResponsibilities")
        if not isinstance(resps, list):
            continue
        for r in resps[:3]:
            if isinstance(r, str) and r.strip():
                key = r.strip().lower()
                if key not in seen:
                    seen.add(key)
                    all_resps.append(r.strip())
    parsed["KeyResponsibilities"] = all_resps[:15]


def _fix_skill_dedup_and_cap(parsed):
    """Deduplicate skills case-insensitively, split into Primary/Secondary, cap at 25."""
    skills = parsed.get("ListOfSkills")
    if not isinstance(skills, list):
        return
    seen = set()
    deduped = []
    for skill in skills:
        if not isinstance(skill, dict):
            continue
        name = skill.get("SkillName", "")
        if not isinstance(name, str) or not name.strip():
            continue
        key = name.strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(skill)

    def sort_key(s):
        months = s.get("SkillExperienceInMonths")
        return months if isinstance(months, (int, float)) and months > 0 else -1
    deduped.sort(key=sort_key, reverse=True)

    parsed["PrimarySkills"] = [s["SkillName"] for s in deduped[:20]]
    parsed["SecondarySkills"] = [s["SkillName"] for s in deduped[20:]]
    parsed["ListOfSkills"] = deduped[:25]


def _fix_cert_validation(parsed):
    """Remove certifications with garbage data (responsibility text, too-long names)."""
    certs = parsed.get("Certifications")
    if not isinstance(certs, list):
        return
    cleaned = []
    for cert in certs:
        if not isinstance(cert, dict):
            continue
        name = cert.get("CertificationName") or cert.get("Name") or ""
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        # Skip if name is too long (likely responsibility text leaked in)
        if len(name) > 100:
            continue
        # Skip if name contains common responsibility verbs
        lower = name.lower()
        if any(verb in lower for verb in ["responsible for", "managed", "developed", "implemented",
                                           "designed", "coordinated", "led the", "worked on"]):
            continue
        cleaned.append(cert)
    parsed["Certifications"] = cleaned


def _fix_current_job_role(parsed):
    """Ensure CurrentJobRole is a proper job title, not a paragraph or responsibility."""
    summary = parsed.get("OverallSummary")
    if not isinstance(summary, dict):
        return
    role = summary.get("CurrentJobRole")
    if not isinstance(role, str) or not role.strip():
        # Try to get from most recent experience
        exps = parsed.get("ListOfExperiences")
        if isinstance(exps, list) and exps:
            first_exp = exps[0]
            if isinstance(first_exp, dict):
                title = first_exp.get("JobTitle")
                if isinstance(title, str) and title.strip():
                    summary["CurrentJobRole"] = title.strip()
        return

    # If role is too long (>80 chars), it's likely responsibility text
    if len(role.strip()) > 80:
        # Extract just the title portion (before comma, dash, or pipe)
        short = re.split(r'[,|\-–—]', role.strip())[0].strip()
        if len(short) > 5 and len(short) <= 80:
            summary["CurrentJobRole"] = short
        else:
            # Fall back to most recent experience title
            exps = parsed.get("ListOfExperiences")
            if isinstance(exps, list) and exps and isinstance(exps[0], dict):
                title = exps[0].get("JobTitle")
                if isinstance(title, str) and title.strip():
                    summary["CurrentJobRole"] = title.strip()


def _fix_responsibility_format(parsed):
    """Split long paragraph responsibilities into sentence-length bullets."""
    experiences = parsed.get("ListOfExperiences")
    if not isinstance(experiences, list):
        return
    for exp in experiences:
        if not isinstance(exp, dict):
            continue
        resps = exp.get("KeyResponsibilities")
        if not isinstance(resps, list):
            continue
        cleaned = []
        for r in resps:
            if not isinstance(r, str) or not r.strip():
                continue
            text = r.strip()
            # If a single bullet is very long (>300 chars), split into sentences
            if len(text) > 300:
                sentences = re.split(r'(?<=[.!?])\s+', text)
                for s in sentences:
                    s = s.strip()
                    if s and len(s) > 10:
                        cleaned.append(s)
            else:
                cleaned.append(text)
        exp["KeyResponsibilities"] = cleaned[:10]  # Cap at 10 per role


def _fix_employment_type(parsed, resume_text=""):
    """Null out FABRICATED default 'Full-time' employment types.

    The LLM often invents "Full-time" when no type is stated. But when the
    resume explicitly says "Full Time" for a role, that's a real value and must
    be kept. Only null a full-time value when the phrase doesn't appear in the
    resume text at all.
    """
    experiences = parsed.get("ListOfExperiences")
    if not isinstance(experiences, list):
        return
    text_lower = (resume_text or "").lower()
    full_time_in_text = "full time" in text_lower or "full-time" in text_lower or "fulltime" in text_lower
    for exp in experiences:
        if not isinstance(exp, dict):
            continue
        emp_type = exp.get("EmploymentType")
        if isinstance(emp_type, str) and emp_type.strip().lower() in _DEFAULT_EMPLOYMENT_TYPES:
            if not full_time_in_text:
                exp["EmploymentType"] = None


def _post_process(parsed, resume_text=""):
    """Orchestrator: apply all post-processing fixes to parsed resume data."""
    if not isinstance(parsed, dict):
        return parsed

    applied = []

    try:
        _fix_name_splitting(parsed)
        applied.append("name_splitting")
    except Exception:
        pass

    try:
        _fix_phone_country_code(parsed)
        applied.append("phone_country_code")
    except Exception:
        pass

    try:
        _fix_languages(parsed)
        applied.append("languages")
    except Exception:
        pass

    try:
        _fix_location_cleanup(parsed)
        applied.append("location_cleanup")
    except Exception:
        pass

    try:
        _fix_experience_years(parsed)
        applied.append("experience_years")
    except Exception:
        pass

    try:
        _fix_total_experience(parsed)
        applied.append("total_experience")
    except Exception:
        pass

    try:
        _fix_skill_experience(parsed)
        applied.append("skill_experience")
    except Exception:
        pass

    try:
        _fix_merge_summary(parsed)
        applied.append("merge_summary")
    except Exception:
        pass

    try:
        _fix_project_fabrication(parsed, resume_text)
        applied.append("project_fabrication")
    except Exception:
        pass

    try:
        _fix_project_company(parsed)
        applied.append("project_company")
    except Exception:
        pass

    try:
        _fix_skill_contamination(parsed)
        applied.append("skill_contamination")
    except Exception:
        pass

    try:
        _fix_empty_certs(parsed)
        applied.append("empty_certs")
    except Exception:
        pass

    try:
        _fix_cert_validation(parsed)
        applied.append("cert_validation")
    except Exception:
        pass

    try:
        _fix_achievements(parsed)
        applied.append("achievements")
    except Exception:
        pass

    try:
        _fix_employment_type(parsed, resume_text)
        applied.append("employment_type")
    except Exception:
        pass

    try:
        _fix_current_job_role(parsed)
        applied.append("current_job_role")
    except Exception:
        pass

    try:
        _fix_responsibility_format(parsed)
        applied.append("responsibility_format")
    except Exception:
        pass

    try:
        _fix_overall_responsibilities(parsed)
        applied.append("overall_responsibilities")
    except Exception:
        pass

    metadata = parsed.get("_metadata")
    if isinstance(metadata, dict):
        metadata["_post_processed"] = applied

    return parsed


def extract_text_from_file(filepath):
    """Extract text from PDF, DOCX, DOC, TXT, or image files."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.pdf':
        return _extract_pdf(filepath)
    elif ext == '.docx':
        return _extract_docx(filepath)
    elif ext == '.doc':
        return _extract_doc(filepath)
    elif ext in ('.jpg', '.jpeg', '.png', '.tiff', '.bmp'):
        return _extract_image_ocr(filepath)
    elif ext in ('.txt', '.html', '.htm'):
        with open(filepath, 'r', errors='ignore') as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _extract_pdf(filepath):
    """Extract text from PDF using PyMuPDF. Falls back to OCR for scanned PDFs."""
    import fitz
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    # If very little text extracted, try OCR (scanned PDF)
    if len(text.strip()) < 50:
        try:
            return _extract_image_ocr(filepath)
        except Exception:
            pass

    return text


def _extract_docx(filepath):
    """Extract text from DOCX."""
    import docx2txt
    return docx2txt.process(filepath)


def _extract_doc(filepath):
    """Extract text from legacy DOC using antiword."""
    import subprocess
    try:
        result = subprocess.run(
            ['antiword', filepath],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except FileNotFoundError:
        pass

    # Fallback: try reading as binary with olefile
    try:
        import olefile
        ole = olefile.OleFileIO(filepath)
        if ole.exists('WordDocument'):
            stream = ole.openstream('WordDocument')
            data = stream.read()
            # Extract ASCII text from binary
            text = data.decode('latin-1', errors='ignore')
            # Filter printable characters
            clean = ''.join(c if c.isprintable() or c in '\n\r\t' else ' ' for c in text)
            ole.close()
            if len(clean.strip()) > 50:
                return clean
    except Exception:
        pass

    raise ValueError("Cannot extract text from DOC file. Install antiword: apt-get install antiword")


def _extract_image_ocr(filepath):
    """Extract text from images using Tesseract OCR."""
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        raise ValueError("OCR not available. Install: pip install pytesseract Pillow")

    ext = os.path.splitext(filepath)[1].lower()

    # For PDFs, convert pages to images first
    if ext == '.pdf':
        import fitz
        doc = fitz.open(filepath)
        full_text = ""
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            from io import BytesIO
            img = Image.open(BytesIO(img_data))
            full_text += pytesseract.image_to_string(img) + "\n"
        doc.close()
        return full_text
    else:
        img = Image.open(filepath)
        return pytesseract.image_to_string(img)


if __name__ == "__main__":
    if not GROQ_API_KEY:
        print("Set GROQ_API_KEY first:")
        print("  export GROQ_API_KEY=gsk_...")
        exit(1)

    test = """
John Smith | john@email.com | (555) 123-4567 | San Francisco, CA
LinkedIn: linkedin.com/in/johnsmith

SUMMARY
Senior Software Engineer with 8 years of experience in full-stack development.

EXPERIENCE
Senior Software Engineer | Google | Jan 2021 - Present
- Led payment system migration to microservices
- Reduced API latency by 40%

Software Engineer | Meta | Jun 2018 - Dec 2020
- Built notification system handling 1M+ events/day

EDUCATION
B.S. Computer Science | Stanford University | 2016 | GPA: 3.8

SKILLS
Python, Java, Go, Docker, Kubernetes, AWS

CERTIFICATIONS
AWS Solutions Architect Professional
"""
    result = parse_resume(test)
    print(json.dumps(result, indent=2))
