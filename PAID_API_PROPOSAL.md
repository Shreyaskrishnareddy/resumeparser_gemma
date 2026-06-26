# Moving the Parsers to Google's Paid API Tier

**Prepared for:** Arytic team
**Re:** Resume Parser & JD Parser — resolving the processing-time concern
**TL;DR:** Switching from the free to the paid Google API tier takes parsing latency from **minutes to seconds**, costs roughly **$1.50 per 1,000 documents (or less)**, requires **no code change and no accuracy change**, and is enabled with a single billing toggle.

---

## 1. Current State & The Problem

Both parsers run on **Google Gemma 4 (31B)** via the Google AI Studio / Gemini API, currently on the **free tier**. The free tier is best-effort shared capacity, which is why QA sees **2–7 minutes per document** and intermittent retries. This is a **throughput limit, not a model or accuracy limit** — extraction quality is unaffected.

## 2. The Paid Approach

Enable **pay-as-you-go billing** on the same Gemini API project. This moves us from free best-effort capacity to the **paid / priority tier**.

- **Same model** (`gemma-4-31b-it`), same prompts, same output schema
- **No code change, no redeployment** — only a billing setting on the API key/project
- **Same accuracy** — already validated (resume ~95%+, JD 97.9% vs QA sheets)

## 3. Cost

Gemma 4 31B is billed at approximately **$0.12 per million input tokens** and **$0.35 per million output tokens**. From our measured usage (~3,500 input + ~3,100 output tokens per document):

**≈ $0.0015 per document (about 0.15 cents).**

| Volume | Cost |
|---|---|
| 1,000 documents | ~$1.50 |
| 5,000 / month | ~$7.50 |
| 10,000 / month | ~$15 |
| 50,000 / month | ~$75 |
| 100,000 / month | ~$150 |

> Note: Google has historically served some Gemma models **free of charge even with billing enabled**. If that applies here, the real cost is **even lower** (latency benefit still applies). We can confirm exact billing in the Google Cloud console after enabling.

## 4. Latency & Throughput

| | Free tier (today) | Paid tier |
|---|---|---|
| Latency / document | 2–7 minutes (variable) | **~5–30 seconds** |
| Rate limit | ~15 requests/min | thousands/min |
| Reliability | intermittent 500s (auto-retried) | priority, stable |

This directly fixes the QA pain point (slow iterative testing) and makes the parsers viable for **real-time, in-flow** use inside the Arytic platform.

## 5. What Changes vs. Stays the Same

| Changes | Stays the same |
|---|---|
| Latency: minutes → seconds | Model (Gemma 4 31B) |
| Rate limits: 15/min → thousands/min | Accuracy & output schema |
| Small per-document cost (~$0.0015) | API endpoints / integration |
| | Hosting (Render), code, deployment |

## 6. Implementation Steps

1. Enable **billing (pay-as-you-go)** on the Google AI Studio / Gemini API project.
2. (Optional) Generate a production API key under the billed project.
3. Update the `GOOGLE_API_KEY` env var on the hosted services — **no code change**.
4. Verify latency on a few test documents.

Lead time: same day.

## 7. Alternatives (for completeness)

- **Google Gemini Flash (paid):** officially supported on Google, also fast and ~similar low cost — a drop-in if we prefer a fully Google-managed model. (Would need a brief re-validation pass.)
- **Third-party hosts** (OpenRouter / DeepInfra / Together) serve the same Gemma 4 at the same token rates — useful as a fallback, but staying on Google is simplest.
- **Self-hosting** the open weights — only worth it at very high, sustained volume.

## Recommendation

**Enable Google's paid tier on the existing Gemini API.** It's the lowest-effort, lowest-risk option: one toggle, no code change, latency drops to seconds, and cost is negligible (~$1.50 per 1,000 documents or less). This unblocks both faster QA cycles and real-time use in production.
