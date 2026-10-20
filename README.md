# AgriSense — Evidence-Grounded Crop Intelligence

AgriSense is a multi-agent crop decision-support prototype with image-based triage, a conservative advisory gate, weather/market context, real-time progress streaming, and saved reports.

> **Safety:** AI results are decision support, not a definitive diagnosis. Do not apply pesticides based on an unverified prediction; use local agricultural extension guidance and follow product labels.

## What is implemented on this branch

- **Real-time FastAPI API** — multipart image upload, WebSocket progress messages, synchronous analysis endpoint, health check, and input-size/image validation.
- **Uncertainty-aware decision gate** — flags unknown/invalid/low-confidence predictions and identifies the current RGB color-statistics heuristic as a demo-only method. Crop-specific advice is suppressed in these cases.
- **Evidence provenance status** — market and climate outputs are labelled unavailable, unverified, or available with supplied provenance.
- **Report persistence and history** — completed reports are written as JSON under `outputs/`; `GET /history` lists recent cases and `GET /cases/{case_id}` retrieves a report.
- **React + Vite dashboard** — drag-and-drop upload, live pipeline timeline, result/uncertainty cards, evidence status, report export, and saved-case history.
- **Automated validation** — Python unit tests, compilation checks, and a frontend build in GitHub Actions.

## Architecture

```text
Crop image
   |
FastAPI upload validation
   |
Real-time session + WebSocket event queue
   |
Orchestrator
   +-- DiagnosisAgent
   +-- Uncertainty gate ---- uncertain/demo output --> conservative next steps
   +-- AdvisoryAgent ------- provisional output ----> provisional guidance
   +-- MarketAgent --------- evidence status/provenance
   +-- ClimateAgent -------- evidence status/provenance
   |
JSON report + local history + dashboard
```

## Research direction: Evidence-Grounded, Uncertainty-Aware Crop Triage (EGUCT)

The proposed system-level contribution combines:

1. **Uncertainty-aware abstention:** don't turn a weak or out-of-scope result into confident treatment advice.
2. **Evidence provenance:** keep source and observation time alongside external data, and show when integrations are not connected.
3. **Safety-oriented orchestration:** gate downstream advice based on diagnosis quality and emit traceable stage events.
4. **Reproducible evaluation:** measure macro-F1, per-class precision/recall, calibration error, abstention coverage/risk, and p50/p95 latency.

This is a research direction, not a claim of proven scientific novelty or increased model accuracy. The strongest academic evidence will come from comparisons on a held-out labelled dataset.

## Important limitations

- The default local diagnosis path includes a green-channel/RGB heuristic. It is **not** a trained disease-specific classifier. The gate marks it as demo-only and suppresses crop-specific treatment recommendations.
- Market and weather agents remain integration adapters. Data is not “live” unless a real source is configured and returns usable data with provenance.
- Model-reported confidence is not independently calibrated by this project yet.
- The memory bank is process-local and resets on restart. Reports are persisted locally, but this is not yet a multi-user production database.
- Crop-specific recommendations require local validation; the system deliberately avoids medication-like certainty.

## Run locally on Windows

Use two terminals from the repository root.

### Terminal 1 — backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn api_realtime:app --reload --host 127.0.0.1 --port 8000
```

Backend docs: http://127.0.0.1:8000/docs

### Terminal 2 — frontend

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal (normally http://127.0.0.1:5173).

The dashboard can set its API location using `frontend/.env.example` as a template. Default API URL is `http://127.0.0.1:8000`.

## Environment variables

Copy `.env.example` to a local `.env`, then fill in only the integrations you actually use. Or set variables directly in the shell:

```powershell
$env:GEMINI_API_KEY = "your-key"
$env:MCP_GATEWAY_URL = "http://localhost:8080"
```

Never commit real keys, passwords, or local environment files.

## API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/` | Health status and active session count |
| POST | `/upload-start` | Start analysis; multipart fields: `file`, `crop_type`, `user_id` |
| WS | `/ws/{session_id}` | Stream structured progress, result, error, and finished events |
| POST | `/analyze` | Run analysis without a WebSocket progress stream |
| GET | `/history?limit=20` | List recent saved reports |
| GET | `/cases/{case_id}` | Retrieve a saved report by UUID |

## Tests

```powershell
python -m compileall -q agents utils api.py api_realtime.py run.py
python -m unittest discover -s tests -v
```

The CI workflow additionally runs `npm install` and `npm run build` for the dashboard.

## Evaluation plan

1. Use a held-out labelled dataset with source/plant separation where possible to reduce leakage.
2. Report a confusion matrix, macro-F1, per-class precision/recall, calibration error, abstention rate, and p50/p95 latency.
3. Compare a base classifier, classifier + uncertainty gate, and full evidence-grounded pipeline.
4. Test blur, low light, non-leaf images, corrupt uploads, stale data, tool outages, and conflicting signals.
5. Record dataset provenance, model version, class mapping, threshold, and experimental settings.
