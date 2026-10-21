# AgriSense — Evidence-Grounded Crop Intelligence

AgriSense is a multi-agent crop decision-support prototype with image-based triage, a conservative advisory gate, weather/market context, real-time progress streaming, and saved reports.

> **Safety:** AI results are decision support, not a definitive diagnosis. Do not apply pesticides based on an unverified prediction; use local agricultural extension guidance and follow product labels.

## What is implemented on this branch

- **Real-time FastAPI API** — multipart image upload, WebSocket progress messages, synchronous analysis endpoint, health check, and input-size/image validation.
- **Uncertainty-aware decision gate** — flags unknown/invalid/low-confidence predictions and identifies the current RGB color-statistics heuristic as a demo-only method. Crop-specific advice is suppressed in these cases.
- **Image acquisition quality gate** — scores basic brightness, contrast, and resolution signals; severe capture issues ask for a new image instead of trusting model output.
- **Live weather context** — location input resolves through Open-Meteo and returns a short-range forecast with transparent heat/rain/wind exposure heuristics.
- **Optional mandi price adapter** — a configurable data.gov.in/Agmarknet resource can return a dated modal price, while unconfigured or unmatched results remain unavailable.
- **Evidence provenance status** — market and climate outputs are labelled unavailable, unverified, or available with supplied provenance; retrieval time is distinguished from source observation time.
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

## Configure live context providers

### Weather

The dashboard's **Farm location** field accepts a city/region name or `latitude,longitude`. The adapter resolves names with the Open-Meteo Geocoding API and requests a short-range forecast from the Open-Meteo Forecast API. These are general weather-exposure heuristics (heat, heavy rain, wind), **not a crop-disease forecast**. The report distinguishes forecast retrieval time from a source observation time. Review the provider's [API documentation](https://open-meteo.com/en/docs) and [geocoding documentation](https://open-meteo.com/en/docs/geocoding-api).

### Mandi market prices (optional)

AgriSense can query an Agmarknet resource through India's [Open Government Data mandi-price dataset](https://www.data.gov.in/resource/current-daily-price-various-commodities-various-markets-mandi). To enable it, register for a data.gov.in API key and select the resource ID that you will query. Add these values to your local `.env`:

```dotenv
DATA_GOV_IN_API_KEY=your_data_gov_in_key
DATA_GOV_IN_MARKET_RESOURCE_ID=your_selected_resource_id
DATA_GOV_IN_PRICE_UNIT=quintal
```

The adapter assumes the configured resource's modal price is per quintal by default and converts it to INR/kg. Set `DATA_GOV_IN_PRICE_UNIT=kg` only when your chosen dataset actually reports prices per kilogram. Resource schemas vary; verify that the selected resource returns the expected commodity, state, market, date, and price columns. A single record does not establish a price trend, so trend remains `unknown` rather than being fabricated.

### Optional LLM / MCP integrations

Copy `.env.example` to a local `.env`, then fill in only the integrations you actually use. Or set variables directly in PowerShell:

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

## Evaluation harness

To compare a baseline classifier with the uncertainty-gated pipeline, create a CSV from the same held-out labelled examples:

```csv
y_true,y_pred,confidence,review_required
leaf_spot,leaf_spot,0.82,false
leaf_blight,unknown,0.43,true
healthy,healthy,0.91,false
```

Run the evaluator from the repository root:

```powershell
python scripts/evaluate_predictions.py --input predictions.csv --output reports/evaluation.json
```

The JSON report includes per-class precision/recall/F1, macro-F1, a confusion matrix, coverage and selective risk after abstentions, Brier score, and expected calibration error (ECE). To support a defensible experiment, compare baseline and gated predictions on identical examples and report dataset splits, class distribution, and model version. The evaluator does not create labels or improve the model by itself.

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
