"""Real-time AgriSense API with progress streaming, validation, and persisted cases.

Run from the repository root:
    python -m uvicorn api_realtime:app --reload --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from PIL import Image, UnidentifiedImageError

from agents.orchestrator import Orchestrator

logger = logging.getLogger("agrisense.realtime")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

MAX_UPLOAD_BYTES = int(os.getenv("AGRISENSE_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_ACTIVE_SESSIONS = int(os.getenv("AGRISENSE_MAX_ACTIVE_SESSIONS", "50"))
SESSION_TTL_SECONDS = int(os.getenv("AGRISENSE_SESSION_TTL_SECONDS", str(30 * 60)))
OUTPUT_DIR = Path(os.getenv("AGRISENSE_OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _create_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.info("GEMINI_API_KEY not configured; optional Gemini backend disabled.")
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception:
        logger.exception("Gemini client could not be initialized; continuing without it.")
        return None


orchestrator = Orchestrator(
    gemini_client=_create_gemini_client(),
    mcp_gateway_url=os.getenv("MCP_GATEWAY_URL"),
)

app = FastAPI(
    title="AgriSense Real-time Crop Intelligence API",
    version="2.0.0",
    description="Crop triage with uncertainty gating, agent progress events, and traceable reports.",
)

origins = [
    item.strip()
    for item in os.getenv(
        "AGRISENSE_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@dataclass
class SessionState:
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    websocket_attached: bool = False


sessions: Dict[str, SessionState] = {}


def _prune_sessions() -> None:
    now = time.time()
    expired = [
        session_id
        for session_id, state in sessions.items()
        if state.completed_at is not None
        and not state.websocket_attached
        and now - state.completed_at > SESSION_TTL_SECONDS
    ]
    for session_id in expired:
        sessions.pop(session_id, None)


def _persist_result(result: Dict[str, Any]) -> Optional[str]:
    case_id = str(result.get("case_id", ""))
    try:
        # case_id is created by the orchestrator, but validate before using it in a path.
        safe_id = str(uuid.UUID(case_id))
        path = OUTPUT_DIR / f"{safe_id}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(jsonable_encoder(result), handle, ensure_ascii=False, indent=2)
        return str(path)
    except Exception:
        logger.exception("Could not persist analysis result")
        return None


async def _run_analysis(
    session_id: str,
    state: SessionState,
    user_id: str,
    crop_type: str,
    image_bytes: bytes,
) -> None:
    loop = asyncio.get_running_loop()

    def progress_callback(step: str, info: Any = None) -> None:
        event: Dict[str, Any] = {"type": "progress", "step": step}
        if isinstance(info, str):
            event["text"] = info
        elif info is not None:
            event["meta"] = jsonable_encoder(info)
        # This callback runs in a worker thread; use the main event loop safely.
        loop.call_soon_threadsafe(state.queue.put_nowait, event)

    try:
        result = await asyncio.to_thread(
            orchestrator.run_case,
            user_id,
            io.BytesIO(image_bytes),
            crop_type,
            progress_callback,
        )
        encoded_result = jsonable_encoder(result)
        saved_path = await asyncio.to_thread(_persist_result, result)
        if saved_path:
            encoded_result["result_file"] = saved_path
        state.queue.put_nowait({"type": "result", "payload": encoded_result})
    except Exception as exc:
        logger.exception("Analysis session %s failed", session_id)
        state.queue.put_nowait({"type": "error", "text": "Analysis failed. Check backend logs for details."})
    finally:
        state.completed_at = time.time()
        state.queue.put_nowait({"type": "finished"})


@app.get("/")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "agrisense-realtime",
        "version": app.version,
        "active_sessions": len(sessions),
    }


@app.get("/history")
async def list_history(limit: int = 20) -> Dict[str, Any]:
    """Return recent locally persisted reports (newest first)."""
    limit = max(1, min(limit, 100))

    def read_summaries():
        reports = []
        for path in OUTPUT_DIR.glob("*.json"):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    report = json.load(handle)
                reports.append({
                    "case_id": report.get("case_id", path.stem),
                    "status": report.get("status", "unknown"),
                    "generated_at": report.get("generated_at"),
                    "diagnosis": report.get("diagnosis", {}).get("label"),
                    "decision_status": report.get("diagnosis_gate", {}).get("status"),
                })
            except (OSError, json.JSONDecodeError):
                logger.warning("Skipping unreadable report %s", path)
        reports.sort(key=lambda item: item.get("generated_at") or "", reverse=True)
        return reports[:limit]

    return {"items": await asyncio.to_thread(read_summaries)}


@app.get("/cases/{case_id}")
async def get_case(case_id: str):
    try:
        safe_id = str(uuid.UUID(case_id))
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "Invalid case ID."})

    path = OUTPUT_DIR / f"{safe_id}.json"
    if not path.is_file():
        return JSONResponse(status_code=404, content={"error": "Case report not found."})

    def read_report():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    try:
        return await asyncio.to_thread(read_report)
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to read case report %s", safe_id)
        return JSONResponse(status_code=500, content={"error": "Could not read case report."})


@app.post("/upload-start")
async def upload_start(
    file: UploadFile = File(...),
    crop_type: str = Form("wheat"),
    user_id: str = Form("farmer_web"),
):
    _prune_sessions()

    if len(sessions) >= MAX_ACTIVE_SESSIONS:
        return JSONResponse(
            status_code=429,
            content={"error": "Too many active analysis sessions. Try again shortly."},
        )

    if file.content_type and not file.content_type.startswith("image/"):
        return JSONResponse(status_code=415, content={"error": "Upload a supported image file."})

    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if not content:
            return JSONResponse(status_code=400, content={"error": "Uploaded file is empty."})
        if len(content) > MAX_UPLOAD_BYTES:
            return JSONResponse(
                status_code=413,
                content={"error": f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit."},
            )
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        return JSONResponse(status_code=400, content={"error": "Invalid or corrupted image file."})
    finally:
        await file.close()

    clean_user_id = user_id.strip()[:120] or "farmer_web"
    clean_crop_type = crop_type.strip()[:80] or "unknown"
    session_id = str(uuid.uuid4())
    state = SessionState()
    sessions[session_id] = state
    asyncio.create_task(
        _run_analysis(session_id, state, clean_user_id, clean_crop_type, content),
        name=f"agrisense-analysis-{session_id}",
    )

    return {
        "session_id": session_id,
        "websocket_url": f"/ws/{session_id}",
        "status": "accepted",
    }


@app.websocket("/ws/{session_id}")
async def stream_session(websocket: WebSocket, session_id: str):
    state = sessions.get(session_id)
    if state is None:
        await websocket.close(code=4404, reason="Unknown or expired analysis session")
        return
    if state.websocket_attached:
        await websocket.close(code=4409, reason="A client is already connected to this session")
        return

    await websocket.accept()
    state.websocket_attached = True
    try:
        while True:
            event = await state.queue.get()
            await websocket.send_json(event)
            if event.get("type") in {"finished"}:
                break
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected from %s", session_id)
    except Exception:
        logger.exception("WebSocket streaming failed for %s", session_id)
    finally:
        state.websocket_attached = False
        if state.completed_at is not None:
            sessions.pop(session_id, None)


@app.post("/analyze")
async def analyze_now(
    file: UploadFile = File(...),
    crop_type: str = Form("wheat"),
    user_id: str = Form("farmer_api"),
):
    """Synchronous API for clients that do not need WebSocket progress."""
    if file.content_type and not file.content_type.startswith("image/"):
        return JSONResponse(status_code=415, content={"error": "Upload an image file."})
    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if not content or len(content) > MAX_UPLOAD_BYTES:
            return JSONResponse(status_code=400, content={"error": "Empty image or upload size limit exceeded."})
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        return JSONResponse(status_code=400, content={"error": "Invalid or corrupted image file."})
    finally:
        await file.close()

    try:
        result = await asyncio.to_thread(
            orchestrator.run_case,
            user_id.strip()[:120] or "farmer_api",
            io.BytesIO(content),
            crop_type.strip()[:80] or "unknown",
        )
        saved_path = await asyncio.to_thread(_persist_result, result)
        encoded = jsonable_encoder(result)
        if saved_path:
            encoded["result_file"] = saved_path
        return encoded
    except Exception:
        logger.exception("Synchronous analysis failed")
        return JSONResponse(status_code=500, content={"error": "Analysis failed. Check backend logs for details."})
