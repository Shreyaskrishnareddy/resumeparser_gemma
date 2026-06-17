# Llama Resume Parser

LLM-powered resume parser using **Llama 3.1 8B** via [Groq](https://groq.com) for fast inference. Parses resumes into structured JSON with high accuracy. Simple resumes parse in 1-3s, complex resumes in 5-10s.

## Features

- **Single-pass full parsing** — no chunking, no token limits, reads the entire resume
- **Structured JSON output** — personal details, experiences, education, skills, certifications, projects, achievements
- **Fast inference** — 1-3s for simple resumes, 5-10s for complex ones via Groq's LPU engine
- **File upload + raw text APIs** — supports PDF, DOC, DOCX, TXT, JPG, PNG, TIFF, BMP (OCR)
- **Web UI included** — drag-and-drop interface with single and bulk upload tabs
- **Async bulk processing** — submit hundreds of resumes, poll progress, download results when done
- **Skill hallucination guard** — post-processing verifies every extracted skill exists in the resume text
- **ATS integration** — output mapped to Bullhorn, Dice, Ceipal formats
- **OCR support** — parse scanned PDFs and images via Tesseract
- **Security headers** — XSS, clickjacking, content-type protections
- **Docker-ready** — one command to deploy anywhere

## Extracted Fields (BRD-Compliant)

| Category | Fields |
|----------|--------|
| Personal Details | FullName, FirstName, MiddleName, LastName, EmailID, PhoneNumber, CountryCode, Location, LinkedIn, GitHub, Portfolio |
| Overall Summary | Summary, CurrentJobRole, RelevantJobTitles (synonyms), TotalExperience, Domain |
| Experience | JobTitle, CompanyName, Location, StartDate, EndDate, ExperienceInYears, Summary, KeyResponsibilities |
| Skills | SkillName, SkillExperienceInMonths, LastUsed, RelevantSkills (synonyms), PrimarySkills, SecondarySkills |
| Education | Degree, TypeOfEducation, Field, Institution, Location, YearPassed, GPA |
| Certifications | CertificationName, IssuerName, IssuedYear |
| Projects | ProjectName, Description, CompanyWorked, RoleInProject, Technologies, StartDate, EndDate, Link |
| Achievements | Quantified accomplishments extracted from entire resume |
| Languages | Language names |

## Quick Start

### Prerequisites

- Python 3.9+
- [Groq API key](https://console.groq.com/keys) (free tier available)

### Setup

```bash
git clone https://github.com/Shreyaskrishnareddy/llama-resumeparser.git
cd llama-resumeparser

pip install -r requirements.txt

export GROQ_API_KEY=gsk_your_key_here

python app.py
```

Open http://localhost:8000 in your browser.

### Docker

```bash
docker build -t llama-resumeparser .

docker run -p 8000:8000 -e GROQ_API_KEY=gsk_your_key_here llama-resumeparser
```

## API Endpoints

### `POST /parse` — Parse a resume file

Upload a PDF, DOCX, or TXT file.

```bash
curl -X POST http://localhost:8000/parse \
  -F "file=@resume.pdf"
```

**Response:**

```json
{
  "filename": "resume.pdf",
  "processing_time_ms": 2100,
  "result": {
    "PersonalDetails": {
      "FullName": "John Smith",
      "Email": "john@email.com",
      "Phone": "(555) 123-4567",
      "Location": "San Francisco, CA",
      "LinkedIn": "linkedin.com/in/johnsmith",
      "GitHub": null,
      "Portfolio": null
    },
    "Summary": "Senior Software Engineer with 8 years of experience...",
    "CurrentJobRole": "Senior Software Engineer",
    "TotalExperience": "8 years",
    "ListOfExperiences": [
      {
        "Company": "Google",
        "Title": "Senior Software Engineer",
        "StartDate": "Jan 2021",
        "EndDate": "Present",
        "Location": "Mountain View, CA",
        "Responsibilities": [
          "Led payment system migration to microservices",
          "Reduced API latency by 40%"
        ]
      }
    ],
    "ListOfEducation": [...],
    "ListOfSkills": ["Python", "Java", "Go", "Docker", "Kubernetes", "AWS"],
    "PrimarySkills": ["Python", "Java", "Go"],
    "SecondarySkills": ["Docker", "Kubernetes", "AWS"],
    "Certifications": ["AWS Solutions Architect Professional"],
    "Projects": [...],
    "Achievements": [...],
    "_metadata": {
      "parser": "groq",
      "model": "llama-3.1-8b-instant",
      "processing_time_ms": 2050,
      "finish_reason": "stop",
      "prompt_tokens": 850,
      "completion_tokens": 1200,
      "total_tokens": 2050
    }
  }
}
```

### `POST /parse/text` — Parse raw text

Send resume text directly as JSON.

```bash
curl -X POST http://localhost:8000/parse/text \
  -H "Content-Type: application/json" \
  -d '{"text": "John Smith\njohn@email.com\n\nEXPERIENCE\n..."}'
```

### `POST /parse/bulk` — Synchronous bulk upload (up to 50 files)

```bash
curl -X POST http://localhost:8000/parse/bulk \
  -F "files=@resume1.pdf" \
  -F "files=@resume2.docx" \
  -F "files=@resume3.jpg"
```

```json
{
  "total_files": 3,
  "successful": 3,
  "failed": 0,
  "total_processing_time_ms": 8500,
  "results": [...]
}
```

### Async Bulk Processing (Recommended for large batches)

#### `POST /jobs/bulk` — Submit async bulk job

Upload files and get a job ID immediately. Files are processed in the background.

```bash
curl -X POST http://localhost:8000/jobs/bulk \
  -F "files=@resume1.pdf" \
  -F "files=@resume2.docx" \
  -F "files=@resume3.pdf"
```

```json
{
  "job_id": "e88b0151138044ebb474a3d66304528c",
  "status": "processing",
  "total_files": 3,
  "message": "Job submitted. Poll GET /jobs/e88b01... for progress."
}
```

#### `GET /jobs/<job_id>` — Poll job progress

```bash
curl http://localhost:8000/jobs/e88b0151138044ebb474a3d66304528c
```

```json
{
  "job_id": "e88b0151138044ebb474a3d66304528c",
  "status": "processing",
  "total_files": 3,
  "completed_files": 2,
  "failed_files": 0,
  "progress_pct": 66.7
}
```

#### `GET /jobs/<job_id>/results` — Download results

Available once job status is `completed`.

```bash
curl http://localhost:8000/jobs/e88b0151138044ebb474a3d66304528c/results
```

```json
{
  "job_id": "e88b01...",
  "total_files": 3,
  "successful": 3,
  "failed": 0,
  "results": [
    { "filename": "resume1.pdf", "status": "completed", "result": { ... } },
    { "filename": "resume2.docx", "status": "completed", "result": { ... } },
    { "filename": "resume3.pdf", "status": "completed", "result": { ... } }
  ]
}
```

### `POST /import/csv` — CSV bulk import

Import candidate records from a CSV file. Each row's columns are sent to the LLM as resume text.

```bash
curl -X POST http://localhost:8000/import/csv \
  -F "file=@candidates.csv"
```

### `POST /parse/ats/<name>` — ATS-formatted output

Parse a resume and return fields mapped to a specific ATS format. Supported: `bullhorn`, `dice`, `ceipal`.

```bash
curl -X POST http://localhost:8000/parse/ats/bullhorn \
  -F "file=@resume.pdf"
```

### `GET /health` — Health check

```bash
curl http://localhost:8000/health
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model to use |
| `PORT` | `8000` | Server port |
| `BULK_RATE_INTERVAL` | `2.0` | Seconds between async bulk API calls |
| `BULK_JOB_TTL_HOURS` | `24` | Hours before completed jobs are cleaned up |
| `BULK_DATA_DIR` | `./data` | Directory for SQLite DB, uploads, and results |

### Supported Models

Any model available on [Groq](https://console.groq.com/docs/models) works. Tested with:

| Model | Speed | Accuracy |
|-------|-------|----------|
| `llama-3.1-8b-instant` | ~2s | High |
| `llama-3.3-70b-versatile` | ~5s | Very High |
| `meta-llama/llama-4-scout-17b-16e-instruct` | ~3s | Very High |

## Project Structure

```
llama-resumeparser/
  app.py              # Flask API server (single, bulk, async, ATS endpoints)
  groq_parser.py      # Core parser — Groq API + JSON extraction + post-processing
  bulk_processor.py   # Async bulk processing — SQLite job queue + background worker
  index.html          # Web UI (single + bulk upload tabs)
  requirements.txt    # Python dependencies
  Dockerfile          # Production container
  .env.example        # Environment variable template
  data/               # Runtime: SQLite DB, uploads, results (gitignored)
```

## How It Works

1. **Text extraction** — PyMuPDF (PDF), docx2txt (DOCX), antiword (DOC), Tesseract OCR (images/scanned PDFs)
2. **LLM parsing** — Full resume text is sent to Llama 3.1 via Groq with a structured JSON schema prompt
3. **JSON extraction** — Response is parsed with a robust extractor that handles markdown fences, whitespace, and malformed output
4. **Structured response** — Clean JSON with all fields + metadata (timing, token usage, model info)

## Deployment

### Render

1. Fork this repo
2. Create a new **Web Service** on [Render](https://render.com)
3. Set environment variable: `GROQ_API_KEY`
4. Build command: `pip install -r requirements.txt`
5. Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`

### Railway / Fly.io / Any Docker Host

```bash
docker build -t llama-resumeparser .
docker run -p 8000:8000 -e GROQ_API_KEY=gsk_... llama-resumeparser
```

## License

MIT
