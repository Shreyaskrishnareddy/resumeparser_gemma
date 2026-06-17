# Gemma 4 Resume Parser

LLM-powered resume parser using **Google Gemma 4** via the [Google AI Studio](https://aistudio.google.com) (Gemini) API. Parses resumes into structured JSON across 43 BRD-compliant fields with high accuracy — reading the entire resume in a single pass, no chunking.

## Features

- **Single-pass full parsing** — no chunking, no token limits; reads the entire resume (Gemma 4's 256K context)
- **Structured JSON output** — personal details, experiences, education, skills, certifications, projects, achievements, languages
- **File upload + raw text APIs** — supports PDF, DOC, DOCX, TXT, HTML, JPG, PNG, TIFF, BMP (OCR)
- **Async bulk processing** — submit hundreds of resumes, poll progress, download results when done
- **Deterministic post-processing** — name splitting, phone/country-code separation, gap-aware total experience, skill de-duplication, skill-hallucination guard, project-fabrication filter
- **Auto-retry** — transient free-tier errors (500/503/429/timeout) are retried with backoff
- **ATS integration** — output mapped to Bullhorn, Dice, Ceipal formats
- **OCR support** — parse scanned PDFs and images via Tesseract
- **Web UI included** — drag-and-drop interface with Structured / JSON / ATS views
- **Docker- and Render-ready** — one command to deploy

## Extracted Fields (BRD-Compliant)

| Category | Fields |
|----------|--------|
| Personal Details | FullName, FirstName, MiddleName, LastName, EmailID, PhoneNumber, CountryCode, Location, LinkedIn, GitHub, Portfolio |
| Overall Summary | Summary, CurrentJobRole, RelevantJobTitles (synonyms), TotalExperience, Domain |
| Experience | JobTitle, CompanyName, Location, StartDate, EndDate, EmploymentType, ExperienceInYears, Summary, KeyResponsibilities |
| Skills | SkillName, SkillExperienceInMonths, LastUsed, RelevantSkills (synonyms), PrimarySkills, SecondarySkills |
| Education | Degree, TypeOfEducation, Field, Institution, Location, YearPassed, GPA |
| Certifications | CertificationName, IssuerName, IssuedYear |
| Projects | ProjectName, Description, CompanyWorked, RoleInProject, Technologies, StartDate, EndDate, Link |
| Achievements | Quantified accomplishments extracted from the entire resume |
| Languages | Spoken language names |

## Quick Start

### Prerequisites

- Python 3.9+
- A free [Google AI Studio API key](https://aistudio.google.com/apikey)

### Setup

```bash
git clone https://github.com/Shreyaskrishnareddy/resumeparser_gemma.git
cd resumeparser_gemma

pip install -r requirements.txt

export GOOGLE_API_KEY=your_key_here

python app.py
```

Open http://localhost:8000 in your browser.

### Docker

```bash
docker build -t resumeparser-gemma .
docker run -p 8000:8000 -e GOOGLE_API_KEY=your_key_here resumeparser-gemma
```

## API Endpoints

### `POST /parse` — Parse a single resume file

```bash
curl -X POST http://localhost:8000/parse -F "file=@resume.pdf"
```

### `POST /parse/text` — Parse raw text

```bash
curl -X POST http://localhost:8000/parse/text \
  -H "Content-Type: application/json" \
  -d '{"text": "John Smith\njohn@email.com\n\nEXPERIENCE\n..."}'
```

### `POST /parse/bulk` — Synchronous bulk upload (up to 50 files)

```bash
curl -X POST http://localhost:8000/parse/bulk \
  -F "files=@resume1.pdf" -F "files=@resume2.docx"
```

### Async Bulk Processing (recommended for large batches)

- `POST /jobs/bulk` — submit files, get a `job_id` immediately
- `GET /jobs/<job_id>` — poll progress
- `GET /jobs/<job_id>/results` — download results once `completed`

### `POST /import/csv` — CSV bulk import

Each row's columns are sent to the model as resume text.

### `POST /parse/ats/<name>` — ATS-formatted output

Supported: `bullhorn`, `dice`, `ceipal`.

```bash
curl -X POST http://localhost:8000/parse/ats/bullhorn -F "file=@resume.pdf"
```

### `GET /health` — Health check

```bash
curl http://localhost:8000/health
# {"status":"healthy","provider":"google","model":"gemma-4-31b-it","configured":true, ...}
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `GOOGLE_API_KEY` | *(required)* | Your Google AI Studio key |
| `GOOGLE_MODEL` | `gemma-4-31b-it` | Gemma model to use |
| `GOOGLE_TIMEOUT` | `300` | Per-request timeout (seconds) |
| `LLM_MAX_RETRIES` | `4` | Retries on transient (500/503/429/timeout) errors |
| `PORT` | `8000` | Server port |
| `BULK_RATE_INTERVAL` | `2.0` | Seconds between async bulk API calls |
| `BULK_JOB_TTL_HOURS` | `24` | Hours before completed jobs are cleaned up |
| `BULK_DATA_DIR` | `./data` | Directory for SQLite DB, uploads, and results |

### Supported Models

Any Gemma model on Google AI Studio works. Tested with:

| Model | Notes |
|-------|-------|
| `gemma-4-31b-it` | 31B dense — highest accuracy (default) |
| `gemma-4-26b-a4b-it` | 26B MoE (4B active) — faster when available |

> **Note on the free tier:** Gemma 4 31B is a large reasoning model; on the free tier a full resume takes ~2–4 minutes and may intermittently return `500`s (best-effort capacity). The built-in retry handles transients; a paid tier removes the latency/variance.

## Project Structure

```
resumeparser_gemma/
  app.py              # Flask API server (single, bulk, async, ATS endpoints)
  gemma_parser.py     # Core parser — Gemma 4 call + JSON extraction + post-processing
  bulk_processor.py   # Async bulk processing — SQLite job queue + background worker
  index.html          # Web UI (Structured / JSON / ATS views)
  requirements.txt    # Python dependencies
  Dockerfile          # Production container
  render.yaml         # Render blueprint
  .env.example        # Environment variable template
  data/               # Runtime: SQLite DB, uploads, results (gitignored)
```

## How It Works

1. **Text extraction** — PyMuPDF (PDF), docx2txt (DOCX), antiword (DOC), Tesseract OCR (images/scanned PDFs)
2. **LLM parsing** — full resume text is sent to Gemma 4 via Google AI Studio with a structured JSON-schema prompt
3. **JSON extraction** — a robust extractor handles reasoning-model prose, markdown fences, smart quotes, and truncated output
4. **Post-processing** — deterministic fixes (name split, dates, skill months, hallucination/fabrication guards) produce clean, grounded fields

## Deployment

### Render

1. Push this repo to GitHub
2. On [Render](https://render.com): **New → Blueprint** → select the repo (uses `render.yaml`)
3. Set `GOOGLE_API_KEY` when prompted (it is `sync: false`)
4. Apply — Render builds and starts it with a 600s worker timeout

### Any Docker Host

```bash
docker build -t resumeparser-gemma .
docker run -p 8000:8000 -e GOOGLE_API_KEY=... resumeparser-gemma
```

## License

MIT
