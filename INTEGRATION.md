# Arytic Parser — Integration Guide

Covers both parser services. They are independent HTTP/JSON microservices.

| Service | Base URL | Repo |
|---|---|---|
| **Resume Parser** | `https://resumeparser-gemma.onrender.com` | `resumeparser_gemma` |
| **JD Parser** | `https://jobparser-gemma.onrender.com` | `jobparser_gemma` |

---

## 1. Tech Stack

- **Language:** Python 3.11
- **Web framework:** Flask + flask-cors, served by **gunicorn** (1 worker, 8 threads)
- **LLM:** Google **Gemma 4 31B** (`gemma-4-31b-it`) via the Google AI Studio (Gemini) API — single-pass extraction, deterministic post-processing
- **Text extraction:** PyMuPDF (PDF), docx2txt (DOCX), antiword (DOC), Tesseract OCR (images/scanned)
- **Hosting:** Docker container on Render (stateless; the resume parser additionally ships an optional SQLite-backed async job queue)
- **Content-Type:** all responses are `application/json` (UTF-8)

## 2. Authentication

Currently **none** (open CORS) — fine for internal/PoC use. For production we recommend **API-key/bearer auth** (`Authorization: Bearer <key>`); we can enable a header check on request.

## 3. Supported Input

PDF, DOC, DOCX, TXT, HTML, JPG, PNG, TIFF, BMP. Limits: **10 MB/file**, **50 files** per bulk request.

## 4. Behaviour & Limits

- **Latency:** Gemma 4 on the **free tier is ~2–7 min/document** (variable best-effort capacity). A **paid Google AI Studio tier** brings this to seconds with no change in accuracy. For volume, use the **async bulk** endpoints so no single HTTP request stays open for minutes.
- **Nulls are intentional:** a field absent from the document returns `null` (never fabricated); empty lists return `[]`.
- **Rate limits (free tier):** ~15 requests/min. Transient upstream `500`s are auto-retried server-side.
- **Error codes:** `400` bad request · `409` async job not yet complete · `404` not found · `502` upstream model error after retries.

---

## 5. Resume Parser — Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/parse` | Parse one uploaded resume file |
| POST | `/parse/text` | Parse raw resume text (`{"text": "..."}`) |
| POST | `/parse/bulk` | Synchronous bulk, up to 50 files |
| POST | `/jobs/bulk` | Submit async bulk job → returns `job_id` (HTTP 202) |
| GET | `/jobs/<job_id>` | Poll async job progress |
| GET | `/jobs/<job_id>/results` | Download results once `status == "completed"` |
| POST | `/import/csv` | Parse candidate rows from a CSV file |
| POST | `/parse/ats/<name>` | Parse + map to an ATS schema (`bullhorn` / `dice` / `ceipal`) |
| GET | `/health` | Liveness + active model |

```bash
# single file
curl -X POST https://resumeparser-gemma.onrender.com/parse -F "file=@resume.pdf"
# raw text
curl -X POST https://resumeparser-gemma.onrender.com/parse/text \
     -H "Content-Type: application/json" -d '{"text":"John Smith ..."}'
# async bulk
curl -X POST https://resumeparser-gemma.onrender.com/jobs/bulk -F "files=@a.pdf" -F "files=@b.docx"
curl https://resumeparser-gemma.onrender.com/jobs/<job_id>
curl https://resumeparser-gemma.onrender.com/jobs/<job_id>/results
```

**Output schema (top-level keys of `result`):** `PersonalDetails`, `OverallSummary`, `ListOfExperiences`, `ListOfSkills`, `PrimarySkills`, `SecondarySkills`, `ListOfEducation`, `Certifications`, `Projects`, `Achievements`, `Languages`, `_metadata`.

---

## 6. JD Parser — Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/parse` | Parse one uploaded JD file |
| POST | `/parse/text` | Parse raw JD text (`{"text": "..."}`) |
| POST | `/parse/bulk` | Synchronous bulk, up to 50 files |
| GET | `/health` | Liveness + active model |

The JD parser returns a richer envelope: every field carries a **confidence score** and **provenance** (character spans back to the source text), plus a document-level `global_confidence`.

---

## 7. Example Response — Resume `POST /parse`

(real output, trimmed: experiences/skills/certs shortened for readability)

```json
{
  "filename": "Ahmad K. Qassem.pdf",
  "processing_time_ms": 134228,
  "result": {
    "PersonalDetails": {
      "FullName": "Ahmad K. Qassem", "FirstName": "Ahmad", "MiddleName": "K.", "LastName": "Qassem",
      "EmailID": "ahmad.elsheikhq@gmail.com", "PhoneNumber": "(312) 723-2889", "CountryCode": "+1",
      "Location": "Austin, TX", "LinkedIn": "linkedin.com/in/ahmadelsheikh",
      "GitHub": "github.com/ahmadelsheikh", "Portfolio": null
    },
    "OverallSummary": {
      "CurrentJobRole": "Project Manager III", "TotalExperience": "10.8 years",
      "Domain": "IT Project Management",
      "RelevantJobTitles": ["IT Project Manager", "Technical Project Manager", "Program Manager"],
      "Summary": "A Postgraduate with ten years of Project Management experience ..."
    },
    "ListOfExperiences": [
      {
        "JobTitle": "Project Manager III", "CompanyName": "United Airlines", "Location": "Remote",
        "StartDate": "July 2021", "EndDate": "Current", "EmploymentType": "Contract",
        "ExperienceInYears": "5.0",
        "KeyResponsibilities": ["Managing high complexity App Dev projects and programs ..."]
      },
      {
        "JobTitle": "Project Manager", "CompanyName": "Emburse", "Location": "CA",
        "StartDate": "Jan 2021", "EndDate": "Jun 2021", "EmploymentType": "Contract",
        "ExperienceInYears": "0.5", "KeyResponsibilities": ["Managed operational workflow ..."]
      }
    ],
    "ListOfSkills": [
      { "SkillName": "SharePoint", "SkillExperienceInMonths": 60, "LastUsed": "Present",
        "RelevantSkills": ["Google Docs", "Google Sheets"] },
      { "SkillName": "MS Project", "SkillExperienceInMonths": 44, "LastUsed": "November 2018",
        "RelevantSkills": ["Planview", "SmartSheet"] }
    ],
    "PrimarySkills": ["SharePoint", "JIRA", "Azure DevOps", "Asana", "Waterfall", "Agile",
                      "Scrum", "PMBOK", "MS Project", "SmartSheet", "Planview", "Android"],
    "SecondarySkills": ["MS Office", "Google Docs", "Google Sheets", "Plan Plus", "iOS"],
    "ListOfEducation": [
      { "Degree": "Master’s Degree", "TypeOfEducation": "Full-time", "Field": "Information Technology",
        "Institution": "Stanford University", "Location": "CA", "YearPassed": "2018", "GPA": null }
    ],
    "Certifications": [
      { "CertificationName": "Project Management Professional (PMP)", "IssuerName": "PMI", "IssuedYear": "2016" },
      { "CertificationName": "Scrum Master Certified", "IssuerName": "Scrum Alliance", "IssuedYear": "2017" }
    ],
    "Projects": [
      { "ProjectName": "Old Post Office site renovation",
        "Description": "A $16.2 million renovation to office space in downtown Chicago ...",
        "CompanyWorked": "PepsiCo", "RoleInProject": "Project Manager",
        "Technologies": ["SmartSheet"], "StartDate": "Aug 2020", "EndDate": "Dec 2020", "Link": null }
    ],
    "Achievements": ["Initiated a $16.2 million renovation to office space at the Old Post Office site"],
    "Languages": ["English", "Arabic"],
    "_metadata": {
      "parser": "gemma", "model": "gemma-4-31b-it", "processing_time_ms": 134228,
      "finish_reason": "STOP", "prompt_tokens": 3469, "completion_tokens": 3127, "total_tokens": 8170
    }
  }
}
```

---

## 8. Example Response — JD `POST /parse`

(real output, trimmed: a few fields shown with full provenance; the rest follow the same `{value, confidence, provenance, status}` shape)

```json
{
  "result": {
    "id": "6ebedf60-88a1-416b-8362-597b591ccd25",
    "source": { "type": "file", "filename": "senior_data_engineer.txt", "url": null,
                "uploaded_at": "2026-06-23T08:20:45Z" },
    "detected_language": "en",
    "global_confidence": 0.944,
    "fields": {
      "title": {
        "value": { "text": "Senior Data Engineer", "seniority_level": "Senior", "domain": "data" },
        "confidence": 1.0,
        "provenance": { "spans": [[11, 31]], "extractor": "job_title_extractor",
                        "extracted_text": "Senior Data Engineer" },
        "status": "ok"
      },
      "company":           { "value": "Austin Energy", "confidence": 1.0, "status": "ok" },
      "location":          { "value": { "city": "Austin", "region": "TX", "country": "USA",
                                        "remote": "hybrid", "formatted_address": "Austin, TX, USA" },
                             "confidence": 0.9, "status": "ok" },
      "employment_type":   { "value": ["contract"], "confidence": 1.0, "status": "ok" },
      "contract_type":     { "value": "C2C", "confidence": 0.95, "status": "ok" },
      "contract_duration": { "value": "12 Months", "confidence": 0.95, "status": "ok" },
      "salary":            { "value": { "min": 75, "max": 75, "currency": "USD", "period": "hour", "ote": false },
                             "confidence": 0.8, "status": "ok" },
      "experience_years":  { "value": { "min_years": 5, "max_years": 8, "requirement_type": "required" },
                             "confidence": 0.95, "status": "ok" },
      "technical_skills":  { "value": ["Python", "SQL", "Spark", "Airflow", "Snowflake", "AWS",
                                       "Glue", "S3", "Redshift", "dbt", "Kafka"],
                             "confidence": 0.95, "status": "ok" },
      "soft_skills":       { "value": ["Collaboration"], "confidence": 0.9, "status": "ok" },
      "job_summary":       { "value": "The Senior Data Engineer will design and maintain scalable data pipelines ...",
                             "confidence": 1.0, "status": "ok" },
      "reporting_to":      { "value": "Data Platform Manager", "confidence": 1.0, "status": "ok" },
      "team_size":         { "value": "6 engineers", "confidence": 1.0, "status": "ok" },
      "benefits":          { "value": ["As per contract (C2C)"], "confidence": 0.9, "status": "ok" }
    },
    "_metadata": { "parser": "gemma_jd_parser", "model": "gemma-4-31b-it",
                   "processing_time_ms": 89959, "finish_reason": "STOP" }
  }
}
```

**JD field list (~35):** `title`, `company`, `location`, `employment_type`, `contract_type`, `contract_duration`, `salary`, `requirements`, `responsibilities`, `skills`, `technical_skills`, `soft_skills`, `education`, `experience_years`, `benefits`, `work_authorization`, `job_domain`, `job_summary`, `description`, `job_id`, `work_mode`, `job_posted_date`, `job_expiry_date`, `reporting_to`, `team_size`, `travel_requirement`, `application_link`, `equal_opportunity_statement`, `company_website`, `industry`, `company_size`, `company_overview`, `preferred_experience`, `preferred_technologies`, `certifications`.

---

## 9. Integration Notes

- **Recommended flow:** single-file `/parse` for interactive use; **async `/jobs/bulk` + poll** for batch ingestion (immune to per-request timeouts).
- **Health check:** poll `GET /health`; expect `{"status":"healthy","model":"gemma-4-31b-it","configured":true}`.
- **Versioning:** the active model and parser are reported in every response's `_metadata`.
- **For production:** request API-key auth + a paid LLM tier (for sub-second latency). We can enable both quickly.
