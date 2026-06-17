# Gemma 4 Resume Parser — Technical Documentation

> LLM-powered resume parser using **Google Gemma 4** via the Google AI Studio (Gemini) API for structured data extraction.
> **Repository**: [github.com/Shreyaskrishnareddy/resumeparser_gemma](https://github.com/Shreyaskrishnareddy/resumeparser_gemma)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Tech Stack](#3-tech-stack)
4. [API Reference](#4-api-reference)
5. [Data Schema](#5-data-schema)
6. [Post-Processing Pipeline](#6-post-processing-pipeline)
7. [Model & Free-Tier Behavior](#7-model--free-tier-behavior)
8. [Deployment](#8-deployment)
9. [Known Limitations](#9-known-limitations)
10. [Configuration Reference](#10-configuration-reference)
11. [Project Structure](#11-project-structure)

---

## 1. Overview

This application parses resumes (PDF, DOCX, DOC, TXT, images) into structured JSON using **Gemma 4 31B** served through the Google AI Studio (Gemini) API. It extracts the 43 BRD fields across 9 categories: personal details, overall summary, work experience (with per-role months), skills (with per-skill experience months), education, certifications, projects, achievements, and languages.

**Why an LLM instead of regex/NLP rules:** traditional parsers break on non-standard formatting. An LLM understands context, so it handles arbitrary layouts, bullet styles, table-based resumes, and date formats. Gemma 4's 256K context lets the entire resume be parsed in a single pass — no chunking, no token-limit failures on long resumes.

**Accuracy model:** the LLM does the extraction; a deterministic post-processing layer then grounds and corrects the output (name splitting, date math, hallucination/fabrication guards). This combination is what keeps fields clean and faithful to the source.

---

## 2. Architecture

```
                Browser / API client
                        |
                index.html  (drag-drop UI: Structured / JSON / ATS)
                        |
          +---------------------------------+
          |        app.py (Flask)           |
          |  /parse  /parse/text /parse/bulk|
          |  /jobs/* (async)  /parse/ats/*  |
          +---------------------------------+
                        |
          +---------------------------------+
          |       gemma_parser.py           |
          |  extract_text_from_file()       |
          |  parse_resume()  -> Gemma call  |
          |  _extract_json() (robust)       |
          |  _post_process() (deterministic)|
          +---------------------------------+
                        |
          +---------------------------------+
          |  Google AI Studio (Gemini API)  |
          |  Model: gemma-4-31b-it          |
          |  generativelanguage.googleapis  |
          +---------------------------------+
```

**Request flow (single parse):**
1. File uploaded to `/parse`; text extracted (PyMuPDF / docx2txt / antiword / Tesseract).
2. Full text injected into a JSON-schema prompt.
3. Single call to Gemma 4 via Google AI Studio (with auto-retry on transient errors).
4. Robust JSON extraction (handles reasoning-model prose, fences, smart quotes, truncation).
5. Deterministic post-processing.
6. Structured JSON returned with a `_metadata` block (model, timing, tokens, finish reason).

**Async bulk** (`bulk_processor.py`): a SQLite-backed job queue with a background daemon thread claims files one at a time, rate-limited, writing a results file when the job completes.

---

## 3. Tech Stack

| Layer | Choice | Role |
|-------|--------|------|
| **API** | Flask + flask-cors | HTTP server, routing |
| **WSGI** | Gunicorn (2 workers, 600s timeout) | Production server; long timeout for multi-minute parses |
| **LLM** | Gemma 4 31B (`gemma-4-31b-it`) | Resume text -> structured JSON |
| **Inference** | Google AI Studio (Gemini API) | Hosted Gemma; 256K context |
| **PDF** | PyMuPDF (fitz) | Text extraction |
| **DOCX** | docx2txt | Text extraction |
| **DOC** | antiword (+ olefile fallback) | Legacy Word extraction |
| **OCR** | Tesseract + Pillow | Scanned PDFs / images |
| **Bulk store** | SQLite (WAL) + threading | Async job queue |

---

## 4. API Reference

See the [README](README.md#api-endpoints) for full examples. Summary:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/parse` | POST | Parse one uploaded file |
| `/parse/text` | POST | Parse raw text (JSON body) |
| `/parse/bulk` | POST | Synchronous bulk (<=50 files) |
| `/jobs/bulk` | POST | Submit async bulk job -> `job_id` |
| `/jobs/<id>` | GET | Poll progress |
| `/jobs/<id>/results` | GET | Download results |
| `/import/csv` | POST | Parse CSV rows as candidates |
| `/parse/ats/<name>` | POST | Bullhorn / Dice / Ceipal mapping |
| `/health` | GET | Health + active model |

Error codes: `400` bad request, `409` job not yet complete, `404` not found, `502` Gemma API error after retries.

---

## 5. Data Schema

Top-level keys: `PersonalDetails`, `OverallSummary`, `ListOfExperiences`, `ListOfSkills`, `PrimarySkills`, `SecondarySkills`, `ListOfEducation`, `Certifications`, `Projects`, `Achievements`, `Languages`, `KeyResponsibilities`, `_metadata`.

`_metadata` example:

```json
"_metadata": {
  "parser": "gemma",
  "model": "gemma-4-31b-it",
  "processing_time_ms": 153582,
  "finish_reason": "STOP",
  "prompt_tokens": 3450,
  "completion_tokens": 1980,
  "total_tokens": 5430,
  "_post_processed": ["name_splitting", "phone_country_code", "..."]
}
```

Convention: missing fields are `null`, missing lists are `[]`. A `null` is a valid, correct value when the data is absent from the resume (e.g. redacted contact info, education years not stated).

---

## 6. Post-Processing Pipeline

After the LLM returns JSON, `_post_process()` applies deterministic fixes (each isolated in try/except so one failure can't break the result):

| Fix | What it does |
|-----|--------------|
| `name_splitting` | Splits FullName into First / Middle / Last by whitespace |
| `phone_country_code` | Separates a leading `+NN` country code into its own field |
| `languages` | Normalizes Languages to clean strings (kills "Objectobject") |
| `location_cleanup` | Strips industry tags / company names out of location fields |
| `experience_years` | Recomputes per-role ExperienceInYears from dates |
| `total_experience` | Total experience = **union of employment intervals** (excludes gaps) |
| `skill_experience` | Per-skill months = sum of jobs whose text mentions the skill; LastUsed = latest such job |
| `merge_summary` | Folds a role Summary into its first responsibility bullet |
| `project_fabrication` | Drops invented projects ("Company + Digital Transformation", single soft-attribute words) |
| `project_company` | Infers a project's company from overlapping experience dates |
| `skill_contamination` | Removes certifications, soft skills, job titles, section headers from skills |
| `cert_validation` | Drops garbage certifications (responsibility text, over-long names) |
| `achievements` | Dedupes, keeps only quantified (number/%/$) achievements, caps at 5 |
| `employment_type` | Nulls only **fabricated** "Full-time" (keeps it when stated in the resume) |
| `current_job_role` | Ensures a clean job title, not paragraph text |
| `responsibility_format` | Splits over-long bullets, caps per role |
| `overall_responsibilities` | Aggregates top responsibilities across roles |

Two grounding guards run after the main pipeline:
- **Skill hallucination guard** — every extracted skill must appear in the resume text, else it is dropped.
- **Skill dedup + cap** — case-insensitive dedup, split into Primary/Secondary, cap at 25.

---

## 7. Model & Free-Tier Behavior

**Default model:** `gemma-4-31b-it` (31B dense). It is a strong, accurate extractor and a reasoning model — it emits internal "thinking" before the JSON, which the extractor handles.

**Free-tier reality (important):**

| Aspect | Behavior |
|--------|----------|
| **Latency** | ~2-4 min per full resume (large model, ~33 tok/s on best-effort capacity) |
| **Context** | 256K — no token-limit failures even on long resumes |
| **Rate limits** | 15 RPM / 1,500 RPD (ample for normal use) |
| **Transient errors** | Occasional `500`s / timeouts from shared free capacity |

The parser **auto-retries** transient errors (`500/503/429`/timeout) with linear backoff (`LLM_MAX_RETRIES`, default 4), which is what lets live free-tier usage succeed despite intermittent failures. A paid Google AI Studio tier removes the latency variance and the 500s.

> The faster `gemma-4-26b-a4b-it` (26B MoE, 4B active) is supported via `GOOGLE_MODEL`, but availability on the free tier is intermittent.

---

## 8. Deployment

**Render (blueprint):** push to GitHub -> **New -> Blueprint** -> select repo (uses `render.yaml`) -> set `GOOGLE_API_KEY` (it is `sync: false`) -> Apply. The blueprint sets `GOOGLE_MODEL`, `GOOGLE_TIMEOUT=300`, `LLM_MAX_RETRIES=4`, and a 600s gunicorn worker timeout.

**Docker:**

```bash
docker build -t resumeparser-gemma .
docker run -p 8000:8000 -e GOOGLE_API_KEY=... resumeparser-gemma
```

The Dockerfile installs `antiword` + `tesseract-ocr` for legacy `.doc` and OCR support.

---

## 9. Known Limitations

| Limitation | Detail / Mitigation |
|------------|---------------------|
| **Free-tier latency** | 2-4 min/resume; use async bulk flow for batches, or a paid tier |
| **Free-tier 500s** | Intermittent best-effort capacity; mitigated by auto-retry |
| **"Present" = today** | An old resume's TotalExperience counts up to the current date |
| **Skill months are inferred** | Estimated from job-description mentions; a skill never named in a role gets `null` months |
| **Legacy `.doc`** | Needs `antiword` (in the Dockerfile); without it, extraction is degraded |

---

## 10. Configuration Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_API_KEY` | Yes | — | Google AI Studio key ([aistudio.google.com/apikey](https://aistudio.google.com/apikey)) |
| `GOOGLE_MODEL` | No | `gemma-4-31b-it` | Gemma model ID |
| `GOOGLE_TIMEOUT` | No | `300` | Per-request timeout (seconds) |
| `LLM_MAX_RETRIES` | No | `4` | Retries on transient errors |
| `MAX_OUTPUT_TOKENS` | No | `12288` | Output token cap |
| `PORT` | No | `8000` | Server port |
| `BULK_RATE_INTERVAL` | No | `2.0` | Seconds between async bulk calls |
| `BULK_JOB_TTL_HOURS` | No | `24` | TTL before completed jobs are cleaned up |
| `BULK_DATA_DIR` | No | `./data` | SQLite DB / uploads / results |

---

## 11. Project Structure

| File | Role |
|------|------|
| `app.py` | Flask API server (single, bulk, async, CSV, ATS endpoints) |
| `gemma_parser.py` | Core: text extraction, Gemma API call, robust JSON extraction, post-processing |
| `bulk_processor.py` | Async bulk: SQLite job queue + background worker |
| `index.html` | Web UI (Structured / JSON / ATS views) |
| `requirements.txt` | Dependencies |
| `Dockerfile` | Production container (antiword + tesseract) |
| `render.yaml` | Render blueprint |
| `.env.example` | Environment template |
