# AgriSense — Autonomous Crop Intelligence

AgriSense is a multi-agent crop decision-support prototype combining image-based crop triage, advisory generation, market context, and climate risk. The goal of the next iteration is to make it **evidence-grounded, uncertainty-aware, testable, and useful in real workflows**—not merely to add more agents.

> **Safety:** AI results are decision support, not a definitive diagnosis. Do not apply pesticides based on an unverified prediction; use local agronomist/extension guidance and follow product labels.

## Architecture

```text
Image upload
   |
FastAPI validation + analysis session
   |
Orchestrator ── progress events ──> React interface
   |
   +── Diagnosis agent (configured classifier/backend)
   +── Evidence/uncertainty gate
   +── Climate agent (configured weather tool)
   +── Market agent (configured market tool)
   +── Advisory agent (grounded recommendations)
   |
Traceable report + session memory
```

## Proposed research contribution: Evidence-Grounded, Uncertainty-Aware Crop Triage (EGUCT)

- **Uncertainty gate:** identify low-confidence and out-of-scope cases; request a clearer image or expert review instead of confidently guessing.
- **Evidence provenance:** attach source and timestamp to weather and market values; explicitly display unavailable data rather than fabricating "live" information.
- **Cross-agent consistency:** check whether the advice is supported by diagnosis and available climate/market evidence; suppress unsupported treatment specifics.
- **Real-time observability:** stream structured stage events to the UI.
- **Reproducible evaluation:** report per-class precision/recall/F1, calibration, abstention rate, and p50/p95 latency.

These are proposed research directions, not claims of proven novelty or improved accuracy. Validate them empirically against a baseline on held-out labeled data.

## Current limitations to address

- The default local classifier includes a green-channel heuristic; it is only a demonstration fallback, not a validated crop-disease model.
- Market and climate tools are optional integrations and can return unavailable/placeholder data.
- The memory bank is process-local and is reset when the app restarts.
- API keys must be stored in environment variables or a secrets manager, never committed.

## Local setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs to test the existing API.

Optional key setup for the current PowerShell session:

```powershell
$env:GEMINI_API_KEY = "your-key"
$env:MCP_GATEWAY_URL = "http://localhost:8080"
```

Never put real secrets into source control.

## API

### `GET /`
Health status.

### `POST /analyze`
Multipart form fields:
- `file`: image
- `crop_type`: crop name (optional; defaults to wheat)
- `user_id`: demo user/session identity (optional)

The real-time API should preserve compatibility with the current orchestrator signature:
`run_case(user_id, image_input, crop_type=None, progress_callback=None)`.
Do not replace this multi-agent call path with a dummy model.

## Evaluation protocol

1. Use a labeled, held-out dataset. Prevent near-duplicate/source leakage between train and test splits.
2. Report macro-F1, per-class precision/recall, confusion matrix, calibration error, coverage-versus-risk under abstention, and p50/p95 response latency.
3. Compare three systems: base diagnosis; diagnosis + uncertainty gate; full evidence-grounded pipeline.
4. Test blur, low light, non-leaf images, corrupt uploads, unavailable tools, stale data, and conflicting evidence.
5. Version the model, dataset provenance, class mapping, and experiment settings.

## Roadmap

1. Align the real-time API and orchestrator interface; add tests.
2. Replace/demo-label heuristic prediction with a trained and evaluated classifier.
3. Add image quality and uncertainty checks.
4. Connect reliable climate/market sources with provenance and freshness.
5. Build a polished dashboard with live progress, evidence cards, explainability, and case history.
6. Publish measured experiments and limitations.
