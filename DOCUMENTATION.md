# Llama Resume Parser — Technical Documentation

> LLM-powered resume parser using Llama 3.1 8B via Groq for structured data extraction.
> **Version**: 1.2.0
> **Last Updated**: 2026-03-20
> **Repository**: [github.com/Shreyaskrishnareddy/llama-resumeparser](https://github.com/Shreyaskrishnareddy/llama-resumeparser)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Tech Stack](#3-tech-stack)
4. [API Reference](#4-api-reference)
5. [Data Schema](#5-data-schema)
6. [Post-Processing Pipeline](#6-post-processing-pipeline)
7. [Model Selection & Tradeoffs](#7-model-selection--tradeoffs)
8. [Rate Limits & Constraints](#8-rate-limits--constraints)
9. [Deployment](#9-deployment)
10. [Testing](#10-testing)
11. [Known Limitations](#11-known-limitations)
12. [Configuration Reference](#12-configuration-reference)
13. [Project Structure](#13-project-structure)
14. [Changelog](#14-changelog)

---

## 1. Overview

### What It Does

This application parses resumes (PDF, DOCX, DOC, TXT, images) into structured JSON using Llama 3.1 8B running on Groq's LPU inference engine. It extracts 11 categories of data: personal details, work experience, education, skills (with per-skill experience months), certifications, projects, achievements, and languages.

### Why This Approach

Traditional resume parsers use regex/NLP rules and break on non-standard formatting. LLM-based parsing handles arbitrary resume layouts, bullet styles, and date formats because the model understands context. Groq's hardware inference makes this fast enough for production (~2-7 seconds per resume).

### How It Works

```
Resume File
    |
    v
Text Extraction (PyMuPDF / docx2txt / antiword / Tesseract OCR)
    |
    v
Full text sent to Llama 3.1 8B via Groq API (single-pass, no chunking)
    |
    v
Robust JSON Extraction (handles markdown fences, malformed output)
    |
    v
Post-Processing (11 deterministic fixes: name splitting, phone/country code separation, ExperienceInYears, SkillExperienceInMonths, merge summary, project company, skill contamination, empty certs, EmploymentType, skill hallucination guard, section header/filler skill filtering)
    |
    v
Structured JSON Response + Metadata
```

---

## 2. Architecture

### System Design

```
                    +------------------+
                    |   Web UI (HTML)  |
                    |  Drag & Drop     |
                    +--------+---------+
                             |
                             v
+----------------------------+----------------------------+
|                     Flask API Server (app.py)            |
|                                                          |
|  /parse          Single file upload                      |
|  /parse/text     Raw text input                          |
|  /parse/bulk     Sync bulk (up to 50, 5 workers)         |
|  /jobs/bulk      Async bulk submit (returns job ID)      |
|  /jobs/<id>      Poll async job progress                 |
|  /jobs/<id>/results  Download async job results          |
|  /import/csv     CSV row-by-row parsing                  |
|  /parse/ats/*    ATS-mapped output (Bullhorn/Dice/Ceipal)|
|  /health         Health check                            |
+----------------------------+----------------------------+
                             |
                             v
+----------------------------+----------------------------+
|               groq_parser.py (Core Logic)                |
|                                                          |
|  extract_text_from_file()  Text extraction dispatcher    |
|  parse_resume()            Groq API call + prompt        |
|  _extract_json()           Robust JSON parser            |
|  _post_process()           Deterministic corrections     |
+----------------------------+----------------------------+
                             |
                             v
              +-----------------------------+
              |   Groq API (External)       |
              |   Model: llama-3.1-8b       |
              |   Endpoint: groq.com/openai |
              +-----------------------------+
```

### Request Flow

1. Client uploads file to `/parse`
2. `extract_text_from_file()` converts to raw text using the appropriate library
3. Full text is injected into the prompt template (no truncation, no chunking)
4. Single API call to Groq with the system prompt + user prompt
5. `_extract_json()` extracts JSON from the LLM response (handles edge cases)
6. `_post_process()` applies deterministic corrections to six known LLM error patterns
7. Metadata (model, timing, tokens, post-processing flags) is attached
8. JSON response returned to client

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| Single-pass (no chunking) | Chunking loses context across sections. Full resume text gives the LLM complete picture. Tradeoff: resumes >16K tokens fail with 413 error. |
| Post-processing over re-prompting | The 8B model consistently makes the same 6 mistakes. Fixing them in Python is faster and more reliable than multi-turn prompting. |
| No database | Stateless API. The parser doesn't store resumes or results. Keeps it simple and avoids PII storage concerns. |
| Gunicorn with 2 workers | Matches Groq's rate limits. More workers would just hit 429 errors. |
| Vanilla JS frontend | No build step, no dependencies. The UI is a single HTML file served by Flask. |

---

## 3. Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM** | Llama 3.1 8B Instant | Resume text → structured JSON |
| **Inference** | Groq API (LPU) | Fast inference (~2s per resume) |
| **Backend** | Flask 3.1 + Gunicorn | API server |
| **Frontend** | Vanilla HTML/CSS/JS | Drag-and-drop web UI |
| **PDF Parsing** | PyMuPDF (fitz) | PDF text extraction |
| **DOCX Parsing** | docx2txt | DOCX text extraction |
| **DOC Parsing** | antiword + olefile | Legacy .doc support |
| **OCR** | Tesseract + Pillow | Scanned PDFs and images |
| **Taxonomy** | groq-taxonomy (shared library) | Skill/title/cert/degree/industry normalization |
| **Deployment** | Docker / Render | Production hosting |

### Dependencies (requirements.txt)

```
Flask==3.1.2
flask-cors==6.0.1
gunicorn==23.0.0
requests==2.32.3
PyMuPDF==1.24.14
docx2txt==0.9
olefile==0.47
pytesseract==0.3.13
Pillow==11.1.0
```

Post-processing uses only stdlib (`calendar`, `re`) — no additional dependencies.

### Shared Library: groq-taxonomy

The project uses `groq-taxonomy`, a shared normalization library from the team monorepo, to enrich parsed resume data with canonical vocabularies.

**Source:** `git+https://github.com/Shreyaskrishnareddy/monorepo.git#subdirectory=packages/taxonomy`
**Version:** 0.1.0
**Dependencies:** None (zero external dependencies)

#### What It Does

After the LLM parses a resume and post-processing runs, `enrich_resume()` normalizes free-text fields to canonical IDs, enabling structured matching between resumes and job descriptions. The import is wrapped in `try/except ImportError` so the parser works without it installed.

```python
from groq_taxonomy import enrich_resume
parsed = enrich_resume(parsed)  # adds _taxonomy key, leaves original fields untouched
```

#### Taxonomy Modules

| Module | Data File | Entries | Key Functions |
|--------|-----------|---------|---------------|
| **skills** | `skills.json` | 329 skills | `normalize_skill(name)` → canonical ID; `classify_skill(id)` → display name, category, subcategory, domain, type, related skills |
| **titles** | `titles.json` | 45 titles + 11 seniority patterns | `normalize_title(text)` → canonical ID, display name, seniority level + weight, function |
| **education** | `degrees.json` + `fields_of_study.json` | 6 degree levels + 33 fields | `normalize_degree(degree, field)` → level, weight, display name, field |
| **certifications** | `certifications.json` | 56 certs | `normalize_cert(text)` → canonical ID, display name, issuer, domain, category |
| **industries** | `industries.json` | 14 industries | `classify_industry(text)` → industry ID; `classify_industry_multi(text)` → ranked list with scores |

#### Text Normalization

The library uses multi-tier text matching (`_text.py`):

- **Standard key** (`make_lookup_key`): lowercased, preserves `#`, `+`, `.` (so "C#" and "C++" stay distinct)
- **Aggressive key** (`make_alias_key`): strips dots/hyphens so "React.js", "reactjs", "react-js" all unify
- **Version stripping**: "Python 3.9" → "python", "Angular 11" → "angular"
- **Suffix variations**: tries "js", ".js", "lang" suffixes for broader matching

#### Skill Categories

Each skill record includes: `id`, `display_name`, `aliases` (e.g., "k8s" → kubernetes), `category` (programming_language, framework, cloud, database, etc.), `subcategory`, `domain` (software_engineering, data_science, etc.), `type` (technical/soft), and `related` skill IDs.

#### Title Seniority Detection

Titles are decomposed into a base role (e.g., `software_engineer`) plus a seniority level with numeric weights:

| Seniority | Weight |
|-----------|--------|
| Intern | 0 |
| Junior | 1 |
| Mid | 2 |
| Senior | 3 |
| Staff | 4 |
| Principal | 5 |
| Distinguished | 6 |
| VP | 7 |
| Director | 8 |
| CTO/CIO | 9 |

#### Degree Levels

| Level | Weight |
|-------|--------|
| High School | 0 |
| Certificate | 1 |
| Associate | 2 |
| Bachelors | 3 |
| Masters | 4 |
| Doctorate | 5 |

#### Enrichment Output

All enrichment data is placed in a `_taxonomy` key on the parsed result, leaving original fields untouched. The enrichment includes:

1. **Skills**: Deduplicated by canonical ID with category/subcategory/domain/type/related metadata
2. **Current job title**: Normalized with seniority detection
3. **Education**: Degree level + field of study normalization
4. **Certifications**: Canonical names with issuer and domain
5. **Industry**: Classification from the domain field
6. **Skill summary**: Counts by category and domain

The library also provides `enrich_jd()` for job description parser output (handles the `{value, confidence, provenance}` envelope format).

---

## 4. API Reference

### Security Headers

All responses include:

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
Cache-Control: no-store
```

---

### `GET /health`

Health check endpoint.

**Response (200):**
```json
{
  "status": "healthy",
  "groq_configured": true,
  "model": "llama-3.1-8b-instant",
  "supported_formats": ["bmp","doc","docx","htm","html","jpeg","jpg","pdf","png","tiff","txt"],
  "max_bulk_files": 50,
  "timestamp": 1772335017.85
}
```

---

### `POST /parse`

Parse a single uploaded resume file.

**Request:** `multipart/form-data` with key `file`

```bash
curl -X POST http://localhost:8000/parse -F "file=@resume.pdf"
```

**Response (200):**
```json
{
  "filename": "resume.pdf",
  "processing_time_ms": 6824,
  "result": {
    "PersonalDetails": { ... },
    "OverallSummary": { ... },
    "ListOfExperiences": [ ... ],
    "ListOfSkills": [ ... ],
    "PrimarySkills": [ ... ],
    "SecondarySkills": [ ... ],
    "ListOfEducation": [ ... ],
    "Certifications": [ ... ],
    "Projects": [ ... ],
    "Achievements": [ ... ],
    "Languages": [ ... ],
    "_metadata": {
      "parser": "groq",
      "model": "llama-3.1-8b-instant",
      "processing_time_ms": 6675,
      "finish_reason": "stop",
      "prompt_tokens": 4933,
      "completion_tokens": 3494,
      "total_tokens": 8427,
      "_post_processed": ["experience_years", "skill_experience", "employment_type"]
    }
  }
}
```

**Error Responses:**

| Code | Cause |
|------|-------|
| 400 | No file, empty filename, unsupported format |
| 502 | Groq API error (rate limit, model error, timeout) |

**Supported Formats:** PDF, DOC, DOCX, TXT, HTML, HTM, JPG, JPEG, PNG, TIFF, BMP
**Max File Size:** 10 MB

---

### `POST /parse/text`

Parse raw resume text (no file upload).

**Request:** `application/json`

```bash
curl -X POST http://localhost:8000/parse/text \
  -H "Content-Type: application/json" \
  -d '{"text": "John Smith\njohn@email.com\n\nEXPERIENCE\nSenior Engineer at Google..."}'
```

**Response:** Same structure as `/parse`, without `filename` wrapper.

**Constraints:** Text must be at least 50 characters.

---

### `POST /parse/bulk`

Parse up to 50 resume files concurrently.

**Request:** `multipart/form-data` with key `files` (multiple)

```bash
curl -X POST http://localhost:8000/parse/bulk \
  -F "files=@resume1.pdf" \
  -F "files=@resume2.docx"
```

**Response (200):**
```json
{
  "total_files": 2,
  "successful": 2,
  "failed": 0,
  "total_processing_time_ms": 14500,
  "results": [ ... ]
}
```

**Constraints:** Max 50 files, 50 MB total. Uses 5 concurrent workers. Blocks until all files are done.

---

### Async Bulk Processing

For large batches, use the async job system. Files are processed in the background with rate limiting (2s between API calls). Progress is trackable via polling.

**Architecture:** SQLite job queue + background daemon thread. Zero external dependencies. Process-safe across Gunicorn workers via WAL mode.

#### `POST /jobs/bulk`

Submit resumes for background processing. Returns immediately with a job ID.

**Request:** `multipart/form-data` with key `files` (multiple)

```bash
curl -X POST http://localhost:8000/jobs/bulk \
  -F "files=@resume1.pdf" \
  -F "files=@resume2.docx"
```

**Response (202):**
```json
{
  "job_id": "e88b0151138044ebb474a3d66304528c",
  "status": "processing",
  "total_files": 2,
  "message": "Job submitted. Poll GET /jobs/e88b01... for progress."
}
```

#### `GET /jobs/<job_id>`

Poll job progress.

**Response (200):**
```json
{
  "job_id": "e88b0151138044ebb474a3d66304528c",
  "status": "processing",
  "total_files": 2,
  "completed_files": 1,
  "failed_files": 0,
  "progress_pct": 50.0,
  "created_at": 1772578608.11,
  "updated_at": 1772578620.55,
  "completed_at": null
}
```

**Status values:** `processing` → `completed`

#### `GET /jobs/<job_id>/results`

Download results when job is completed.

**Response (200):**
```json
{
  "job_id": "e88b01...",
  "total_files": 2,
  "successful": 2,
  "failed": 0,
  "results": [
    { "filename": "resume1.pdf", "status": "completed", "processing_time_ms": 6712, "result": { ... } },
    { "filename": "resume2.docx", "status": "completed", "processing_time_ms": 6368, "result": { ... } }
  ]
}
```

**Error (409):** Returned if job is not yet completed.

#### Job Lifecycle

```
POST /jobs/bulk → [processing] → background thread picks up files one by one
  → each file: pending → processing → completed/failed
  → job counters updated after each file
  → [completed] → results JSON written to data/results/{job_id}.json
  → [cleaned up after 24h] → uploads, results, DB records deleted
```

#### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `BULK_RATE_INTERVAL` | `2.0` | Seconds between Groq API calls (~30 req/min) |
| `BULK_JOB_TTL_HOURS` | `24` | Hours before completed jobs are cleaned up |
| `BULK_DATA_DIR` | `./data` | Directory for SQLite DB, uploads, and results |

---

### `POST /import/csv`

Import candidate records from CSV. Each row's columns are concatenated as resume text.

```bash
curl -X POST http://localhost:8000/import/csv -F "file=@candidates.csv"
```

**Constraints:** Max 50 rows per import.

---

### `POST /parse/ats/<ats_name>`

Parse resume and return fields mapped to a specific ATS system.

**Supported ATS:** `bullhorn`, `dice`, `ceipal`

```bash
curl -X POST http://localhost:8000/parse/ats/bullhorn -F "file=@resume.pdf"
```

**Response (200):**
```json
{
  "filename": "resume.pdf",
  "ats": "bullhorn",
  "processing_time_ms": 7200,
  "data": {
    "firstName": "Ahmad",
    "lastName": "Elsheikh",
    "email": "ahmad@gmail.com",
    "phone": "+1 312 723 2889",
    "occupation": "Project Manager III",
    "skillList": ["MS Project", "JIRA", "SharePoint"]
  },
  "full_result": { ... }
}
```

**ATS Field Mappings:**

| Bullhorn | Dice | Ceipal | Source Field |
|----------|------|--------|-------------|
| firstName | - | FirstName | PersonalDetails.FirstName |
| lastName | - | LastName | PersonalDetails.LastName |
| email | email_address | Email | PersonalDetails.EmailID |
| phone | phone_number | Phone | PersonalDetails.PhoneNumber |
| occupation | current_title | JobTitle | OverallSummary.CurrentJobRole |
| skillList | skills | Skills | PrimarySkills / ListOfSkills |
| - | work_history | - | ListOfExperiences |
| educationDegree | education | Education | ListOfEducation |

---

## 5. Data Schema

### Full Output Structure

```json
{
  "PersonalDetails": {
    "FullName": "string",
    "FirstName": "string",
    "MiddleName": "string | null",
    "LastName": "string",
    "EmailID": "string | null",
    "PhoneNumber": "string | null",
    "CountryCode": "string (e.g. +1)",
    "Location": "string | null",
    "LinkedIn": "string | null",
    "GitHub": "string | null",
    "Portfolio": "string | null"
  },
  "OverallSummary": {
    "Summary": "string",
    "CurrentJobRole": "string",
    "RelevantJobTitles": ["string"],
    "TotalExperience": "string (e.g. 10 years)",
    "Domain": "string"
  },
  "ListOfExperiences": [
    {
      "JobTitle": "string",
      "CompanyName": "string",
      "Location": "string | null",
      "StartDate": "string (e.g. July 2021)",
      "EndDate": "string (e.g. Present)",
      "EmploymentType": "string (Contract/Part-time/Internship) | null",
      "ExperienceInYears": "string (e.g. 4.6)",
      "Summary": "string",
      "KeyResponsibilities": ["string"]
    }
  ],
  "ListOfSkills": [
    {
      "SkillName": "string",
      "SkillExperienceInMonths": "integer | null",
      "LastUsed": "string (e.g. February 2026) | null",
      "RelevantSkills": ["string"]
    }
  ],
  "PrimarySkills": ["string"],
  "SecondarySkills": ["string"],
  "ListOfEducation": [
    {
      "Degree": "string",
      "TypeOfEducation": "string (Full-time/Part-time/Online/Distance)",
      "Field": "string",
      "Institution": "string",
      "Location": "string | null",
      "YearPassed": "string | null",
      "GPA": "string | null"
    }
  ],
  "Certifications": [
    {
      "CertificationName": "string",
      "IssuerName": "string | null",
      "IssuedYear": "string | null"
    }
  ],
  "Projects": [
    {
      "ProjectName": "string",
      "Description": "string",
      "CompanyWorked": "string | null",
      "RoleInProject": "string | null",
      "Technologies": ["string"],
      "StartDate": "string | null",
      "EndDate": "string | null",
      "Link": "string | null"
    }
  ],
  "Achievements": ["string or object"],
  "Languages": ["string or object"],
  "_metadata": {
    "parser": "groq",
    "model": "string",
    "processing_time_ms": "integer",
    "finish_reason": "string",
    "prompt_tokens": "integer",
    "completion_tokens": "integer",
    "total_tokens": "integer",
    "_post_processed": ["string"]
  }
}
```

### Field Rules (enforced via prompt)

| Rule | Details |
|------|---------|
| Name splitting | Last word = LastName, first word = FirstName, middle = MiddleName |
| CountryCode | Always prefixed with "+" (e.g. "+1", "+91") |
| EmploymentType | Only if explicitly stated in resume. null otherwise. Never defaults to "Full-time". |
| Skills | Only explicitly written skills. No certifications, no spoken languages, no soft skills. |
| Languages | Only spoken/human languages. Programming languages go to Skills. |
| Achievements | Max 5 items, must contain quantified metrics from the resume. Never fabricated. |
| KeyResponsibilities | ALL bullets from resume included. Never truncated. |
| CurrentJobRole | Must be from the most recent experience entry, not a section header. |

---

## 6. Post-Processing Pipeline

### Why Post-Processing Is Needed

The Llama 3.1 8B model produces good structured JSON but consistently makes several types of errors that cannot be fixed through prompt engineering alone:

| Problem | LLM Behavior | Example |
|---------|-------------|---------|
| **Name splitting** | Inconsistent First/Middle/Last splitting | "Ahmad Qassem Ahmad Elsheikh" split incorrectly |
| **Phone/CountryCode** | Country code embedded in PhoneNumber | PhoneNumber: "+1 (312) 723-2889" instead of separated |
| **ExperienceInYears** | Math errors in date calculations | July 2021 → Present = "5.2" (should be 4.6) |
| **SkillExperienceInMonths** | Fabricates identical values for all skills | All skills get 120 or 180 months |
| **Skill contamination** | Section headers and filler terms extracted as skills | "Plan Plus", "Learning", "Technical Tools" in PrimarySkills |
| **Summary duplication** | Summary text not merged into responsibilities | Summary and first bullet contain same text |
| **Project company** | Links projects to companies unreliably | CompanyWorked filled with wrong company |
| **EmploymentType** | Defaults to "Full-time" even when not stated | Every role gets "Full-time" |

### Solution

Deterministic Python post-processing in `_post_process()` runs after every successful LLM parse. Each fix is wrapped in try/except so one failure doesn't block others.

### Fix 1: `_fix_name_splitting()`

**What:** Deterministically splits `FullName` into `FirstName`, `MiddleName`, and `LastName` by whitespace.

**How:**
1. Splits `FullName` on whitespace
2. First word → `FirstName`, last word → `LastName`, everything in between → `MiddleName`
3. For 2-word names, `MiddleName` = null
4. For 1-word names, both `FirstName` and `LastName` are set to that word
5. All parts are title-cased for consistency

**Why needed:** The LLM sometimes splits names inconsistently, especially with multi-part names from different cultural conventions. Deterministic splitting ensures the "last word = LastName" rule is always applied.

### Fix 2: `_fix_phone_country_code()`

**What:** Separates country code from phone number into the `CountryCode` field.

**How:**
1. Detects leading `+` prefix followed by 1-3 digits and a separator in `PhoneNumber`
2. Strips the country code portion and stores it in `CountryCode`
3. Ensures `CountryCode` always has the `+` prefix (fixes bare digits like "1" → "+1")
4. Preserves existing `CountryCode` if already populated

**Example:**
| Field | Before | After |
|-------|--------|-------|
| PhoneNumber | +1 (312) 723-2889 | (312) 723-2889 |
| CountryCode | null | +1 |

### Fix 3: `_fix_experience_years()`

**What:** Recalculates `ExperienceInYears` from `StartDate` and `EndDate` for every experience entry.

**How:**
1. `_parse_date()` converts LLM date strings to `(year, month)` tuples
   - Handles: "July 2021", "Jan 2020", "04/2015", "2021", "2020-08", "Present", "Current"
   - "Present" = current date (dynamic)
   - Year-only dates default to January
2. `_calc_months()` computes months between two tuples (minimum 1)
3. Result is `round(months / 12, 1)` stored as a string like "4.6"

**Verified Results (Ahmad Qasem):**

| Role | StartDate | EndDate | LLM Output | Corrected |
|------|-----------|---------|------------|-----------|
| United Airline | July 2021 | Present | 5.2 | **4.6** |
| Emburse | Jan 2021 | Jun 2021 | 0.8 | **0.4** |
| PepsiCo | Aug 2020 | Dec 2020 | 0.5 | **0.3** |

### Fix 4: `_fix_skill_experience()`

**What:** Recalculates `SkillExperienceInMonths` by searching each experience's text for skill name mentions.

**How:**
1. Pre-computes a list of `(searchable_text, duration_months, end_date)` for each experience entry
   - `searchable_text` = Summary + all KeyResponsibilities joined, lowercased
2. For each skill, does a case-insensitive substring search across all experiences
3. Sums months for matching experiences
4. Skills not found in any experience text get `null`
5. `LastUsed` is updated to the end date of the most recent matching role
6. Skills with names shorter than 2 characters are skipped (avoids false positives)

**Known Limitation:** Substring matching can produce false positives (e.g., "Go" matches "Django"). This is an accepted tradeoff for simplicity. Skills with very short names (1 char) are skipped entirely.

**Verified Results (Mutchie):**

| Skill | LLM Output | Corrected | Reason |
|-------|-----------|-----------|--------|
| Linux | 180 | **205** | Mentioned across many roles |
| Solaris | 180 | **48** | Only in Sun-era roles |
| VDI | 180 | **48** | Only in Oracle/Sun roles |
| Python | 180 | **null** | In skills section only, not in experience text |
| FORTRAN | 120 | **null** | In skills section only |

### Fix 5: `_fix_merge_summary()`

**What:** Merges each experience entry's `Summary` field into `KeyResponsibilities` as the first bullet, then clears the `Summary`.

**How:**
1. For each experience entry with a non-empty `Summary`
2. Checks if the first `KeyResponsibilities` bullet already matches the summary (avoids duplication)
3. If not a duplicate, inserts the summary as the first bullet
4. Sets `Summary` to null

**Why needed:** The LLM sometimes duplicates the summary text both in `Summary` and as the first responsibility bullet. This fix consolidates everything into `KeyResponsibilities` for a cleaner output, since the ATS downstream consumes responsibilities as the primary content.

### Fix 6: `_fix_project_company()`

**What:** Removes `CompanyWorked` from all project entries by setting it to null.

**How:** Iterates through all projects and sets `CompanyWorked` to null.

**Why needed:** The LLM often incorrectly links projects to companies based on proximity in the resume text rather than actual association. Since the linkage is unreliable, it's safer to remove it than to present wrong data.

### Fix 7: `_fix_employment_type()`

**What:** Nulls out "Full-time", "Full Time", and "Fulltime" values that the LLM fabricates when the resume doesn't state employment type.

**How:** Simple string comparison against a set of default values. "Contract", "Part-time", "Internship", "Freelance", "Temporary" are kept as-is.

**Verified Results (Ahmad Qasem):**

| Role | LLM Output | Corrected |
|------|-----------|-----------|
| EtQ | Full Time | **null** |
| United Airline | Contract | **Contract** (kept) |

### Fix 8: `_fix_skill_contamination()`

**What:** Removes certifications, soft skills, job titles, section headers, and generic filler terms that leaked into `ListOfSkills`.

**How:** Checks each skill's name against five keyword sets:
1. **Certifications** — PMP, Scrum Master, CCNA, AWS Certified, etc.
2. **Soft skills** — communication, leadership, teamwork, problem solving, etc.
3. **Job titles** — terms containing manager, engineer, developer, analyst, etc.
4. **Section headers** — "Technical Tools", "Application Software", "Operating Systems", "Programming Languages", "Core Competencies", etc. (34 entries, exact match)
5. **Generic filler terms** — "plan plus", "learning", "lessons learned", "planning", "training", "best practices", etc. (20 entries, exact match)

Also filters skill names longer than 60 characters (likely responsibility text).

**Why exact match for headers/fillers:** Prevents false positives — "Machine Learning" is kept while "Learning" alone is filtered. "Operating Systems" the header is removed while "Linux" passes through.

**QA-verified results:**
| Resume | Before | After |
|--------|--------|-------|
| Ahmad K. Qassem | "Plan Plus", "Lessons Learned" in PrimarySkills | Removed |
| Zaman S | "Technical Tools", "Application Software" in skills | Removed |
| All 19 QA resumes | Various contamination | 0 bad skills across all resumes |

### Fix 9: `_fix_empty_certs()`

**What:** Removes hallucinated placeholder certification objects with null/empty names.

**How:** Filters out any certification where `CertificationName` is empty, "null", "none", or "n/a".

**Why needed:** The LLM sometimes generates empty cert objects to fill the schema template.

### Fix 10: `_fix_skill_hallucination()` (text verification)

**What:** Removes skills whose `SkillName` does not appear in the actual resume text.

**How:**
1. Takes both the parsed JSON and the original resume text
2. For each skill, checks if the name appears in the resume using smart matching:
   - Special characters (C#, .NET, C++) → direct substring match
   - Short names (C, R, Go) → word-boundary regex
   - Standard names → case-insensitive substring
3. Skills not found in the resume text are removed

**Why needed:** The LLM sometimes generates plausible-sounding skills that aren't actually written in the resume (e.g., "Angular 11", "Python 3", "Financial Process Optimization"). This fix eliminated 396 out of 409 mismatches in deep verification testing.

**Verified Results (Lakshman Podili):** 217 → 28 skills (removed 189 hallucinated Azure sub-service variants)
**Verified Results (Zaman S):** 227 → 50 skills (removed 177 repetitive financial process variants)

### Metadata

All post-processing results are tracked in `_metadata._post_processed`:

```json
"_metadata": {
  "_post_processed": ["name_splitting", "phone_country_code", "languages", "location_cleanup", "experience_years", "total_experience", "skill_experience", "merge_summary", "project_company", "skill_contamination", "empty_certs", "cert_validation", "achievements", "employment_type", "current_job_role", "responsibility_format", "overall_responsibilities"]
}
```

The skill hallucination fix runs separately after `_post_process()` since it requires access to the original resume text.

If a fix fails (bad data), it's silently skipped and omitted from the list.

---

## 7. Model Selection & Tradeoffs

### Models Evaluated

We tested three models on the Groq free tier for resume parsing:

| Model | Params | TPM | RPD | TPD | Speed |
|-------|--------|-----|-----|-----|-------|
| **llama-3.1-8b-instant** | 8B | 6K | 14.4K | 500K | ~2-7s |
| llama-3.3-70b-versatile | 70B | 12K | 1K | 100K | ~5s |
| llama-4-scout-17b-16e-instruct | 17B MoE | 30K | 1K | 500K | ~3s |

### Why We Chose llama-3.1-8b-instant

**We tested `llama-4-scout-17b` against `llama-3.1-8b` on the same resume (Ahmad Qasem).** Results:

| Metric | 8B | Scout 17B | Winner |
|--------|-----|-----------|--------|
| Skills extracted | 20 | 5 | **8B** |
| PrimarySkills populated | Yes | Empty | **8B** |
| Projects extracted | 1 | 0 | **8B** |
| KeyResponsibilities per role | 14, 10, 10, 9, 13 | 13, 9, 9, 8, 11 | **8B** |
| FullName accuracy | "Ahmad Qassem Ahmad Elsheikh" | "Ahmad Elsh eikh" (broken) | **8B** |
| Tokens used | 8,427 | 7,102 | Scout |
| Parse time | 6.8s | 6.4s | Scout |
| EmploymentType hallucination | Yes (fixed by post-processing) | Less frequent | Scout |

**Verdict:** The 8B model extracts significantly more data. Its three weaknesses (ExperienceInYears math, SkillExperienceInMonths fabrication, EmploymentType defaults) are fully corrected by our post-processing pipeline. The Scout model is faster and has higher TPM limits but misses too much data for an ATS use case where completeness matters.

**The 70B model** would give the best quality but has severe rate limits: 1K RPD and 100K TPD, making it impractical for any real workload.

### Tradeoff Summary

```
Chose: Completeness of extraction (8B) + deterministic post-processing
Over:  Smarter model (Scout/70B) with less post-processing needed

Reasoning:
- An ATS parser that misses 75% of skills is worse than one that needs math corrections
- Post-processing is cheap (microseconds, no API calls)
- 8B has 14.4K RPD vs Scout's 1K RPD (14x more daily requests)
```

---

## 8. Rate Limits & Constraints

### Groq Free Tier Limits (llama-3.1-8b-instant)

| Limit | Value | Impact |
|-------|-------|--------|
| **TPM** (tokens/min) | 6,000 | Main bottleneck. A single resume uses ~5K-9K tokens. Effectively 1 resume/minute. |
| **RPM** (requests/min) | 30 | Not a bottleneck in practice. |
| **RPD** (requests/day) | 14,400 | Sufficient for moderate workloads. |
| **TPD** (tokens/day) | 500,000 | ~55-100 resumes/day depending on length. |

### Request Size Limits

| Constraint | Value | Impact |
|-----------|-------|--------|
| Max tokens per request | ~6K prompt | Resumes over ~16K characters hit 413 errors |
| Max file size | 10 MB | Set in Flask config |
| Max bulk files | 50 | Per bulk upload request |
| Bulk total size | 50 MB | All files combined |
| CSV max rows | 50 | Per CSV import |
| API timeout | 60 seconds | Per Groq API call |

### Test Suite Results (22 resumes)

| Category | Count | Percentage |
|----------|-------|-----------|
| Successfully parsed | 7 | 32% |
| 413 Too Large (>16K tokens) | 9 | 41% |
| 429 Rate Limited | 5 | 23% |
| JSON Parse Failure | 2 | 9% |
| Text Extraction Error | 1 | 5% |

**Key Insight:** The 6K TPM limit is the primary constraint. 41% of resumes exceeded the 8B model's context window. Upgrading to Groq's Developer tier ($0) raises TPM to 20K-60K and enables larger resumes.

---

## 9. Deployment

### Render (Current)

The project deploys to Render via `render.yaml`:

```yaml
services:
  - type: web
    name: llama-resumeparser
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
    envVars:
      - key: GROQ_API_KEY
        sync: false
      - key: GROQ_MODEL
        value: llama-3.1-8b-instant
    healthCheckPath: /health
```

**Steps:**
1. Push code to GitHub
2. Create Web Service on [dashboard.render.com](https://dashboard.render.com)
3. Connect repository, select **Python** runtime
4. Set `GROQ_API_KEY` environment variable
5. Change health check path to `/health`
6. Deploy

**Note:** Free tier instances spin down after inactivity. First request after idle will take 30-60 seconds (cold start). Use Starter ($7/mo) for always-on.

### Docker

```bash
docker build -t llama-resumeparser .
docker run -p 8000:8000 -e GROQ_API_KEY=gsk_your_key llama-resumeparser
```

The Dockerfile includes system dependencies (antiword for .doc, tesseract for OCR).

### Local Development

```bash
pip install -r requirements.txt
export GROQ_API_KEY=gsk_your_key
python app.py
# Server starts on http://localhost:8000 with debug mode
```

---

## 10. Testing

### Test Suite (`test_resumes.py`)

The test suite:
1. Iterates through 22 test resumes (PDF, DOCX, DOC)
2. Extracts raw text from each file
3. Sends each file to the `/parse` API endpoint
4. Cross-verifies every extracted field against the raw resume text
5. Generates an Excel report with per-field pass/fail/warn results

**Running Tests:**
```bash
# Ensure the API server is running on port 8000
python app.py &

# Run the test suite (takes ~10 min due to rate limits)
python -u test_resumes.py
```

**Output:** `test-results/Resume_Parser_Test_Report.xlsx` with two sheets:
- **Summary** — per-resume status (PASS/FAIL/WARN/ERROR), check counts, issues
- **Field Details** — every individual field check with extracted value, result, and notes

**Rate Limit Delay:** 25 seconds between API calls (configured in `RATE_LIMIT_DELAY`).

### Manual Verification

Post-processing was manually verified on two resumes:

1. **Ahmad Qasem** — ExperienceInYears corrected (5.2→4.6, 0.8→0.4, 0.5→0.3), EmploymentType "Full Time"→null, Contract values preserved
2. **Mutchie** — All 9 "Full-time" values nulled, SkillExperienceInMonths varies per skill (was all 180), ExperienceInYears recalculated for all 9 roles

---

## 11. Known Limitations

### Model Limitations

| Limitation | Details | Mitigation |
|-----------|---------|------------|
| **Context window** | Resumes >16K chars get 413 errors | Upgrade to Groq Developer tier or use a larger model |
| **JSON reliability** | ~10% of parses return malformed JSON | `_extract_json()` handles markdown fences, brace matching. Retry on failure. |
| **Name parsing** | PDF text extraction sometimes introduces spaces mid-word | None currently — depends on PDF quality |
| **Hallucination** | Model may infer skills or job titles not in the resume | Prompt engineering reduces this but doesn't eliminate it |

### Post-Processing Limitations

| Limitation | Details |
|-----------|---------|
| **Substring matching** | "Go" matches "Django", "C" matches "Cisco". Short skill names cause false positives. |
| **Year-only dates** | "2018" defaults to January 2018. Could be any month. |
| **Present date hardcoded** | "Present" = February 2026. Must be updated over time. |
| **Skills only matched in KeyResponsibilities + Summary** | Skills used in a role but not mentioned in bullets won't be matched. |

### Infrastructure Limitations

| Limitation | Details |
|-----------|---------|
| **Free tier TPM** | 6K tokens/min = ~1 resume per minute |
| **No persistence** | Parsed results are not stored. Client must save the response. |
| **No authentication** | API is open. Add auth middleware for production. |
| **No retry logic** | Rate limit errors (429) are returned directly to the client. |
| **No .doc support without antiword** | Legacy .doc files require antiword system package. |

---

## 12. Configuration Reference

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | — | Groq API key from [console.groq.com/keys](https://console.groq.com/keys) |
| `GROQ_MODEL` | No | `llama-3.1-8b-instant` | Model ID to use for parsing |
| `PORT` | No | `8000` | Server port |

### Prompt Configuration

The system prompt and parse prompt are defined as constants in `groq_parser.py`:
- `SYSTEM_PROMPT` — Sets the LLM's role as an expert resume parser
- `PARSE_PROMPT` — Contains all extraction rules + JSON schema template

**LLM Parameters:**
```python
temperature: 0.1    # Low randomness for consistent structured output
top_p: 0.9
max_tokens: 8192    # Max response length
timeout: 60s        # Per-request timeout
```

---

## 13. Project Structure

```
llama-resumeparser/
|
+-- app.py                  # Flask API server (single, bulk, async, ATS endpoints)
+-- groq_parser.py          # Core logic (text extraction, LLM call, JSON parsing, post-processing)
+-- bulk_processor.py       # Async bulk processing (SQLite job queue + background worker)
+-- index.html              # Web UI (single + bulk upload tabs, progress tracking)
+-- requirements.txt        # Python dependencies
+-- Dockerfile              # Docker build (Python 3.11 + antiword + tesseract)
+-- render.yaml             # Render deployment config
+-- .env.example            # Environment variable template
+-- .gitignore              # Git ignore rules
+-- LICENSE                 # MIT License
+-- README.md               # Project README
+-- DOCUMENTATION.md        # This file
|
+-- deep_verify.py          # Deep field-level verification (42 fields × 22 resumes)
+-- test_resumes.py         # Test suite (22 resumes, cross-verification, Excel report)
+-- test-results/           # Test output (JSON results, Excel reports)
+-- data/                   # Runtime: SQLite DB, uploads, results (gitignored)
```

### File Responsibilities

| File | Lines | Responsibility |
|------|-------|---------------|
| `groq_parser.py` | ~1060 | Text extraction, Groq API integration, JSON extraction, 17-step post-processing pipeline |
| `app.py` | ~430 | HTTP endpoints, file handling, async job routes, ATS field mapping, security, CORS |
| `bulk_processor.py` | ~250 | SQLite job store, background daemon thread, rate-limited processing, auto-cleanup |
| `index.html` | ~530 | Web UI with single/bulk tabs, progress bar, results table, JSON download |
| `deep_verify.py` | ~640 | 42-field verification against resume text, Excel report generation |

---

## 14. Changelog

### v1.2.0 (2026-03-20)

**Skill Contamination Fix**
- Extended `_fix_skill_contamination()` with two new filter sets: section headers (34 entries) and generic filler terms (20 entries)
- Filters resume table headers like "Technical Tools", "Application Software", "Operating Systems" that were being extracted as skills
- Filters generic non-technical terms like "Plan Plus", "Learning", "Lessons Learned", "Planning", "Training"
- Uses exact match to avoid false positives ("Machine Learning" kept, "Learning" removed)
- QA result: 19/19 resumes pass with zero skill contamination

**Phone/Country Code Separation**
- Added `_fix_phone_country_code()` post-processing step
- Strips leading country code (+1, +91, +44, etc.) from PhoneNumber into CountryCode field
- Ensures CountryCode always has `+` prefix
- QA result: 19/19 resumes pass with properly separated fields

**Prompt Optimization**
- Trimmed PARSE_PROMPT from ~38 rules to ~16 rules (~50% fewer prompt tokens)
- Removed instructions for fields handled by post-processing (name splitting, experience calculation, employment type, location cleanup)
- Reduced `max_completion_tokens` from 32768 to 12288
- Added stronger conciseness constraints for KeyResponsibilities

**Date Handling**
- Fixed hardcoded "Present" date — now uses `date.today()` dynamically instead of static (2026, 2)

### v1.1.0 (2026-03-03)

**Async Bulk Processing**
- Added `bulk_processor.py` — SQLite-backed job queue with background daemon thread
- New endpoints: `POST /jobs/bulk`, `GET /jobs/<id>`, `GET /jobs/<id>/results`
- Rate-limited processing (2s between API calls, configurable via `BULK_RATE_INTERVAL`)
- Auto-cleanup of expired jobs (default 24h TTL)
- Process-safe across Gunicorn workers via SQLite WAL mode

**Skill Hallucination Guard**
- Added `_fix_skill_hallucination()` — verifies every SkillName against actual resume text
- Smart matching: word-boundary regex for short names (C, R, Go), direct substring for special chars (C#, .NET, C++)
- Eliminated 396/409 mismatches in deep verification (97% reduction)

**Additional Post-Processing**
- Added `_fix_skill_contamination()` — removes certifications and soft skills from ListOfSkills
- Added `_fix_empty_certs()` — removes hallucinated placeholder cert objects

**Web UI**
- Added tabbed interface: Single Resume / Bulk Upload
- Bulk tab: multi-file selection, folder upload, drag & drop
- Live progress bar with polling
- Results table with name, role, skills count, status per file
- "View" button to inspect individual results, "Download JSON" for export

**Deep Verification**
- Created `deep_verify.py` — verifies all 42 official data fields across all resumes
- Status system: MATCH, PARTIAL, MISMATCH, NULL (null = acceptable, not failure)
- Generates 4-sheet Excel report matching team's data fields format
- Final results: 99.7% accuracy (MATCH+PARTIAL), 0.3% MISMATCH rate

### v1.0.0 (2026-02-28)

**Post-Processing Pipeline**
- Added `_fix_experience_years()` — recalculates ExperienceInYears from StartDate/EndDate
- Added `_fix_skill_experience()` — computes SkillExperienceInMonths from experience text search
- Added `_fix_employment_type()` — nulls fabricated "Full-time" defaults
- Added `_post_process()` orchestrator with per-fix error isolation
- Added `_parse_date()` flexible date parser (7 formats + "Present")
- Added `_metadata._post_processed` tracking array

**Prompt Improvements**
- Added explicit ExperienceInYears calculation instructions with worked examples
- Added EmploymentType rule: null if not stated, never default to "Full-time"
- Added SkillExperienceInMonths rule: per-skill calculation, no identical values
- Added rules for: name splitting, certifications vs skills, spoken vs programming languages
- Added FINAL REMINDERS section enforcing all critical rules

**Infrastructure**
- Fixed `app.py` health endpoint to use `GROQ_MODEL` constant instead of separate hardcoded default
- Model evaluation: tested llama-4-scout-17b vs llama-3.1-8b, kept 8B for extraction completeness

### v0.x (prior)

- Initial Groq resume parser with Llama 3.1 8B
- Flask API with single file, bulk, CSV, and ATS endpoints
- Web UI with drag-and-drop
- Docker + Render deployment support
- DOC/OCR support, security headers
