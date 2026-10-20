"""Orchestrate crop diagnosis, market/climate context, and conservative advisory.

Public API:
    run_case(user_id, image_input, crop_type=None, progress_callback=None)

progress_callback receives progress_callback(step: str, info: str | dict | None).
Callback failures are isolated so frontend telemetry cannot crash an analysis.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional
import logging
import time
import uuid

try:
    from google.adk.sessions import InMemorySessionService  # type: ignore
except Exception:
    class InMemorySessionService:
        """Small local fallback for demos when ADK is not installed."""
        def __init__(self):
            self.sessions = {}

        def start_session(self, user_id: str):
            sess = type("SessionObj", (), {})()
            sess.user_id = user_id
            sess.started_at = time.time()
            sess.context = {}

            def to_dict():
                return {
                    "user_id": user_id,
                    "started_at": sess.started_at,
                    "context": sess.context,
                }

            sess.to_dict = to_dict
            self.sessions[user_id] = sess
            return sess

        def get_session(self, user_id: str):
            return self.sessions.get(user_id)


from tools.mcp_tool import MCPToolClient
from utils.memory_bank import MemoryBank
from utils.logging_tracing import log_event
from utils.decision_guard import assess_diagnosis
from agents.diagnosis_agent import DiagnosisAgent
from agents.advisory_agent import AdvisoryAgent
from agents.market_agent import MarketAgent
from agents.climate_agent import ClimateAgent

logger = logging.getLogger("agrisense.orchestrator")
logger.setLevel(logging.INFO)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _annotate_evidence(kind: str, value: Any) -> Dict[str, Any]:
    """Mark whether tool data is available and whether provenance accompanies it."""
    checked_at = _utc_now()
    if not isinstance(value, dict):
        value = {"value": value}

    result = dict(value)
    note = str(result.get("note", "")).lower()
    error_present = bool(result.get("error"))
    placeholder_markers = (
        "no market_tool", "no weather_tool", "not connected",
        "placeholder", "not configured", "connect an openapi",
    )
    placeholder = any(marker in note for marker in placeholder_markers)

    if kind == "market":
        missing_core_data = (
            result.get("avg_price_per_kg") is None
            and str(result.get("trend", "unknown")).lower() == "unknown"
        )
    else:
        missing_core_data = str(result.get("risk", "unknown")).lower() == "unknown"

    if error_present or placeholder or missing_core_data:
        status = "unavailable"
    elif result.get("source") and (
        result.get("observed_at") or result.get("fetched_at") or result.get("timestamp")
    ):
        status = "available_with_provenance"
    else:
        status = "unverified"

    observed_at = (
        result.get("observed_at")
        or result.get("fetched_at")
        or result.get("timestamp")
    )
    result["evidence"] = {
        "kind": kind,
        "status": status,
        "source": result.get("source"),
        "observed_at": observed_at,
        "checked_at": checked_at,
        "freshness": "unknown" if not observed_at else "timestamp_provided_not_independently_verified",
    }
    return result


def _conservative_advice() -> str:
    return (
        "The system cannot safely recommend a crop-specific treatment from this result. "
        "Take a clear, well-lit photo of the affected leaf (including the whole leaf and close-up), "
        "record the crop and symptoms, and consult a local agricultural extension officer. "
        "Avoid applying or mixing pesticides based only on an AI prediction."
    )


class Orchestrator:
    def __init__(
        self,
        *,
        session_service: Optional[Any] = None,
        mcp_gateway_url: Optional[str] = None,
        gemini_client: Optional[Any] = None,
        local_model: Optional[Any] = None,
        market_tool: Optional[Any] = None,
        weather_tool: Optional[Any] = None,
        min_diagnosis_confidence: Optional[float] = None,
    ):
        self.session_service = session_service or InMemorySessionService()
        self.memory = MemoryBank()
        self.mcp = MCPToolClient(gateway_url=mcp_gateway_url)
        self.gemini = gemini_client
        self.local_model = local_model
        self.min_diagnosis_confidence = min_diagnosis_confidence

        self.diagnosis = DiagnosisAgent(
            mcp_tool=self.mcp,
            gemini_client=self.gemini,
            local_model=self.local_model,
        )
        self.advisory = AdvisoryAgent(gemini_client=self.gemini)
        self.market = MarketAgent(market_tool=market_tool)
        self.climate = ClimateAgent(weather_tool=weather_tool)

    def run_case(
        self,
        user_id: str,
        image_input,
        crop_type: str = None,
        progress_callback: Optional[Callable[[str, Any], None]] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the agents and emit structured progress events in pipeline order."""
        started = time.monotonic()
        case_id = str(uuid.uuid4())
        config = config or {}

        def emit(step: str, info: Any = None):
            if progress_callback is not None:
                try:
                    progress_callback(step, info)
                except Exception:
                    logger.debug("Progress callback failed at %s", step, exc_info=True)

        emit("case.started", {"case_id": case_id, "user_id": user_id})
        log_event("orchestrator.start", user_id=user_id, case_id=case_id)

        try:
            session = self.session_service.start_session(user_id)
        except Exception as exc:
            logger.warning("Session creation failed; using minimal session: %s", exc)
            session = type("SessionObj", (), {})()
            session.user_id = user_id
            session.context = {}
            session.to_dict = lambda: {"user_id": user_id, "context": {}}

        stored_memory = self.memory.get_user(user_id) or {}
        user_memory = dict(stored_memory)
        # Location and market region are explicit per-case inputs, not inferred from account metadata.
        for context_key in ("location", "market_region"):
            supplied_value = config.get(context_key)
            if isinstance(supplied_value, str) and supplied_value.strip():
                user_memory[context_key] = supplied_value.strip()[:200]

        # 1. Image-based diagnosis
        emit("diagnosis.started", {"message": "Validating image and running configured diagnosis backends"})
        try:
            diagnosis = self.diagnosis.run(image_input, crop_type=crop_type)
            if not isinstance(diagnosis, dict):
                diagnosis = {"label": "unknown", "confidence": None, "error": "Invalid diagnosis payload"}
        except Exception as exc:
            logger.exception("Diagnosis failed for case %s", case_id)
            diagnosis = {"label": "error", "confidence": 0.0, "error": str(exc)}

        min_confidence = config.get(
            "min_diagnosis_confidence",
            self.min_diagnosis_confidence,
        )
        gate = assess_diagnosis(diagnosis, min_confidence=min_confidence)
        if diagnosis.get("needs_expert_review"):
            gate["review_required"] = True
            gate["allows_crop_specific_advice"] = False
            gate["status"] = "review_required"
            gate["reasons"].append("diagnosis_backend_requested_expert_review")
            gate["message"] = (
                "The image model flagged this result for expert review. Capture a clearer image "
                "and consult a local agricultural extension officer before selecting a treatment."
            )

        emit("diagnosis.completed", {
            "label": diagnosis.get("label", "unknown"),
            "reported_confidence": gate.get("reported_confidence"),
            "decision_status": gate["status"],
            "review_required": gate["review_required"],
        })
        log_event(
            "diagnosis.result",
            user_id=user_id,
            case_id=case_id,
            label=diagnosis.get("label"),
            confidence=gate.get("reported_confidence"),
            decision_status=gate["status"],
        )

        # 2. Market context, before advisory so it can be included as evidence.
        emit("market.started", {"message": "Checking configured market-data integration"})
        try:
            crop_for_market = crop_type or diagnosis.get("crop") or "unknown"
            market_region = user_memory.get("market_region")
            market_info = self.market.get_market_insights(crop_for_market, region=market_region)
        except Exception as exc:
            logger.warning("Market agent failed for case %s: %s", case_id, exc)
            market_info = {"error": str(exc), "note": "Market integration unavailable"}
        market_info = _annotate_evidence("market", market_info)
        emit("market.completed", {"evidence_status": market_info["evidence"]["status"]})

        # 3. Climate context, before advisory.
        emit("climate.started", {"message": "Checking configured weather integration"})
        try:
            climate_info = self.climate.assess_location_risk(location=user_memory.get("location"))
        except Exception as exc:
            logger.warning("Climate agent failed for case %s: %s", case_id, exc)
            climate_info = {"error": str(exc), "note": "Weather integration unavailable"}
        climate_info = _annotate_evidence("climate", climate_info)
        emit("climate.completed", {"evidence_status": climate_info["evidence"]["status"]})

        # 4. Advisory receives the context results, but a review-required diagnosis blocks
        # crop-specific advice regardless of its reported numeric confidence.
        if gate["allows_crop_specific_advice"]:
            emit("advisory.started", {"message": "Generating provisional guidance using available evidence"})
            try:
                advice = self.advisory.run({
                    "diagnosis": diagnosis,
                    "user_memory": user_memory,
                    "crop": crop_type,
                    "market": market_info,
                    "climate": climate_info,
                })
                advisory_mode = "agent_generated_provisional"
            except Exception as exc:
                logger.warning("Advisory agent failed for case %s: %s", case_id, exc)
                advice = _conservative_advice()
                advisory_mode = "conservative_fallback"
        else:
            emit("advisory.skipped", {
                "message": "Crop-specific treatment suppressed because diagnosis needs review"
            })
            advice = _conservative_advice()
            advisory_mode = "uncertainty_guard"
        emit("advisory.completed", {"mode": advisory_mode})

        duration_ms = round((time.monotonic() - started) * 1000, 2)
        report = {
            "case_id": case_id,
            "status": "completed",
            "generated_at": _utc_now(),
            "diagnosis": diagnosis,
            "diagnosis_gate": gate,
            "advice": advice,
            "advisory_mode": advisory_mode,
            "safety_note": (
                "AI output is decision support, not a confirmed diagnosis. Verify locally before treatment "
                "and follow official product-label instructions."
            ),
            "market": market_info,
            "climate": climate_info,
            "session": session.to_dict() if hasattr(session, "to_dict") else {},
            "run_metadata": {
                "pipeline": "agrisense-evidence-gated-v1",
                "duration_ms": duration_ms,
                "confidence_calibrated": gate.get("confidence_calibrated", False),
                "config": {
                    "min_diagnosis_confidence": gate.get("minimum_confidence"),
                },
            },
        }

        user_memory.update({
            "last_case_id": case_id,
            "last_diag": diagnosis,
            "last_advice": advice,
            "market": market_info,
            "climate": climate_info,
            "last_case_at": report["generated_at"],
        })
        try:
            self.memory.save_user(user_id, user_memory)
        except Exception as exc:
            logger.warning("Saving case memory failed for %s: %s", user_id, exc)

        log_event(
            "orchestrator.end",
            user_id=user_id,
            case_id=case_id,
            duration_ms=duration_ms,
        )
        emit("case.completed", {"case_id": case_id, "duration_ms": duration_ms})
        return report


if __name__ == "__main__":
    print("Example: Orchestrator().run_case('demo-user', 'leaf1.png', crop_type='wheat')")
