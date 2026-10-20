# AgriSense — Autonomous Crop Intelligence Platform

AgriSense is evolving from a multi-agent crop-diagnosis demo into an **evidence-grounded crop decision-support system**. It combines image-based crop triage, explainable uncertainty, weather context, market context, and a traceable advisory workflow.

> **Important:** Model outputs and external data are decision support, not a substitute for a local agronomist. Do not apply pesticides from an unverified AI prediction; follow local labels and extension guidance.

## Current architecture

- **Diagnosis agent** — runs the configured image-classification backend. A simple color heuristic may be used as a demo fallback; it is not a validated disease classifier.
- **Advisory agent** — generates a structured response or conservative rule-based guidance.
- **Climate agent** — uses Open-Meteo geocoding/forecast APIs when a farm location is supplied; risk flags are transparent general weather heuristics, not disease predictions.
- **Market agent** — optionally queries a configured data.gov.in/Agmarknet resource; reports unavailable when credentials/resource/schema are missing.
- **Orchestrator** — coordinates agents, stores short-lived user context, and emits structured progress events.
- **FastAPI** — serves synchronous analysis and real-time job updates.
- **Image-quality gate** — flags severely dark/overexposed, extremely low-resolution, or nearly uniform captures before permitting crop-specific advisory.
- **React UI** — provides upload, image-quality feedback, live agent timeline, results, evidence status, report export, and case history.

## Proposed next-level novelty

**Evidence-Grounded, Uncertainty-Aware Crop Triage (EGUCT)**

1. **Image-quality-aware abstention:** use interpretable capture checks and the diagnosis uncertainty gate to trigger a retake or expert review instead of unsupported treatment claims.
2. **Evidence provenance:** keep source observation time separate from retrieval time; include provider/source and forecast target dates. Missing integrations remain visibly unavailable.
3. **Evidence-contextual advisory:** inject the current diagnosis, weather summary, and available market data into the advisory prompt while forbidding invented prices, weather, and pesticide dosage.
4. **Real-time trace:** stream ordered pipeline events, store report JSON, and expose case-history endpoints.
5. **Reproducible evaluation:** measure per-class precision/recall/F1, calibration, image-quality abstention, coverage-versus-risk, p50/p95 latency, and unsupported-claim rate.

These are engineering/research hypotheses to evaluate, not claims of proven novelty or improved accuracy. The strongest academic contribution will come from testing this pipeline against a baseline on a labeled dataset.

## Repository setup

Use a virtual environment and install:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Set optional API keys in environment variables. Never commit secrets:

```powershell
$env:GEMINI_API_KEY = "your-key"
```

Start the real-time API from the repository root:

```powershell
python -m uvicorn api_realtime:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal, start the dashboard:

```powershell
cd frontend
npm install
npm run dev
```

API docs: http://127.0.0.1:8000/docs

## API

### `GET /`
Health check.

### `POST /analyze`
Multipart form fields:
- `file` — image file
- `crop_type` — crop name (optional; defaults to wheat)
- `user_id` — session identity for the demo (optional)

### Real-time mode
The `api_realtime.py` endpoint streams progress and accepts multipart fields `file`, `crop_type`, `user_id`, `location`, and `market_region`. The orchestrator contract is `run_case(user_id, image_input, crop_type=None, progress_callback=None, config=None)`. Do not replace the multi-agent path with a dummy classifier; demo fallback is explicitly identified.

## Validation plan

- Evaluate on a held-out, labeled dataset with train/test separation by source/plant where possible.
- Report precision, recall, macro-F1, confusion matrix, expected calibration error, abstention coverage, and latency (p50/p95).
- Compare: baseline diagnosis; diagnosis + uncertainty gate; full evidence-grounded multi-agent pipeline.
- Test corrupt uploads, non-leaf images, low-light/blurry images, unavailable API integrations, and contradictory signals.
- Record dataset provenance, class distribution, model version, and experiment configuration for reproducibility.

## Limitations

- Current market adapter and weather adapter may be unconfigured; do not label their output “live” until connected to a reliable source.
- A color heuristic is only a demo fallback, not a substitute for a trained and evaluated classifier.
- The memory bank is in-process and resets on restart; use persistent storage for deployment.
- Real-world pesticide advice requires validated crop/disease/locality information and label-compliant guidance.

## Roadmap

1. Stabilize one API contract and test the full image-to-report flow.
2. Connect a validated classifier and add uncertainty/quality checks.
3. Connect reliable, attributable weather and market data sources.
4. Upgrade UI with live progress, evidence cards, recommendations, and case history.
5. Run the evaluation plan and publish measured results.
