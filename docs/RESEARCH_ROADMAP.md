# AgriSense — Autonomous Crop Intelligence Platform

AgriSense is evolving from a multi-agent crop-diagnosis demo into an **evidence-grounded crop decision-support system**. It combines image-based crop triage, explainable uncertainty, weather context, market context, and a traceable advisory workflow.

> **Important:** Model outputs and external data are decision support, not a substitute for a local agronomist. Do not apply pesticides from an unverified AI prediction; follow local labels and extension guidance.

## Current architecture

- **Diagnosis agent** — runs the configured image-classification backend. A simple color heuristic may be used as a demo fallback; it is not a validated disease classifier.
- **Advisory agent** — generates a structured response or conservative rule-based guidance.
- **Climate agent** — accepts a configured weather tool and reports its coverage/availability.
- **Market agent** — accepts a configured market-data tool and reports its coverage/availability.
- **Orchestrator** — coordinates agents, stores short-lived user context, and emits structured progress events.
- **FastAPI** — serves synchronous analysis and real-time job updates.
- **React UI** — intended to show upload, agent timeline, results, evidence freshness, and case history.

## Proposed next-level novelty

**Evidence-Grounded, Uncertainty-Aware Crop Triage (EGUCT)**

1. **Uncertainty gate:** distinguish a confident prediction from uncertain, out-of-scope, or poor-quality inputs. When confidence is weak, request a better image or expert review instead of inventing a disease.
2. **Evidence provenance:** attach source, observation time, location/crop scope, and freshness to weather/market signals. Missing integrations are explicitly marked unavailable—not presented as live.
3. **Cross-agent consistency:** compare diagnosis, weather risk, and advisory claims; surface contradictions and suppress unsupported treatment details.
4. **Real-time trace:** stream structured events per analysis session to the UI.
5. **Reproducible evaluation:** measure per-class precision/recall/F1, calibration, abstention rate, latency, and recommendation grounding.

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

Start the existing API:

```powershell
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
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
The `api_realtime.py` endpoint can provide progress streaming. Keep its API contract aligned with `Orchestrator.run_case(user_id, image_input, crop_type=None, progress_callback=None)`. Do not replace the project orchestrator with a dummy classifier; use the configured diagnosis agent and clearly label any demo fallback.

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
