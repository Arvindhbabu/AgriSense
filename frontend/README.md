# AgriSense Dashboard

React + Vite UI for the AgriSense real-time API.

## Run locally

Terminal 1 — from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn api_realtime:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL printed in the terminal (usually http://127.0.0.1:5173).

Optional: copy `.env.example` to `.env` to change `VITE_API_BASE_URL`.
The API allows localhost:5173 and localhost:3000 by default.

## Included UI

- Drag-and-drop image upload and local preview
- Farm location input for live Open-Meteo forecast context
- Optional market-state filter for configured data.gov.in/Agmarknet records
- Image acquisition quality warnings and uncertainty-aware review state
- Live WebSocket event timeline
- Uncertainty/review status and reported confidence
- Evidence status for market and climate integrations
- Saved report history and JSON export

The dashboard intentionally displays unavailable or unverified external data as such. The default RGB heuristic is a demo fallback, not a clinically/field-validated disease classifier.
