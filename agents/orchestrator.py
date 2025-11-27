# agents/orchestrator.py
"""
Orchestrator: simple agent that coordinates Diagnosis -> Advisory -> Market -> Climate
Designed to be called from a notebook, CLI runner, or small API.

This version includes a safe fallback for the ADK InMemorySessionService so the
project runs locally even when `google.adk` is not installed.
"""
from typing import Dict, Any, Optional
import logging
import time

# Try to import ADK session service; if unavailable provide a small local fallback
try:
    from google.adk.sessions import InMemorySessionService  # type: ignore
except Exception:
    # Lightweight fallback session service for local/demo runs.
    # It provides start_session(user_id) -> session object with .to_dict()
    class InMemorySessionService:
        def __init__(self):
            self.sessions = {}

        def start_session(self, user_id: str):
            """
            Create a minimal session object with a `to_dict()` method expected by Orchestrator.
            """
            # simple session object
            sess = type("SessionObj", (), {})()
            sess.user_id = user_id
            sess.started_at = time.time()
            sess.context = {}
            def to_dict():
                return {"user_id": user_id, "started_at": sess.started_at, "context": sess.context}
            sess.to_dict = to_dict
            # store for potential retrieval
            self.sessions[user_id] = sess
            return sess

        def get_session(self, user_id: str):
            return self.sessions.get(user_id)

from tools.mcp_tool import MCPToolClient
from utils.memory_bank import MemoryBank
from utils.logging_tracing import log_event

# import local agents
from agents.diagnosis_agent import DiagnosisAgent
from agents.advisory_agent import AdvisoryAgent
from agents.market_agent import MarketAgent
from agents.climate_agent import ClimateAgent

logger = logging.getLogger("agrisense.orchestrator")
logger.setLevel(logging.INFO)


class Orchestrator:
    def __init__(
        self,
        *,
        session_service: Optional[Any] = None,
        mcp_gateway_url: Optional[str] = None,
        gemini_client: Optional[Any] = None,
        local_model: Optional[Any] = None
    ):
        """Initialize orchestrator with optional clients and services.

        session_service: object exposing start_session(user_id) -> session
        mcp_gateway_url: URL for MCP gateway (if used)
        gemini_client: LLM / Vision client, optional
        local_model: local TF/PyTorch model object, optional
        """
        # Session & memory
        self.session_service = session_service or InMemorySessionService()
        self.memory = MemoryBank()

        # Tools / clients
        self.mcp = MCPToolClient(gateway_url=mcp_gateway_url)
        self.gemini = gemini_client
        self.local_model = local_model

        # Agents
        self.diagnosis = DiagnosisAgent(mcp_tool=self.mcp, gemini_client=self.gemini, local_model=self.local_model)
        self.advisory = AdvisoryAgent(gemini_client=self.gemini)
        self.market = MarketAgent(market_tool=None)  # set tool or leave None to use placeholders
        self.climate = ClimateAgent(weather_tool=None)  # set tool or leave None

    def run_case(self, user_id: str, image_input, crop_type: str = None) -> Dict[str, Any]:
        """
        Top-level flow:
         1) Start session / fetch memory
         2) Diagnose image
         3) Produce advisory
         4) Get market insights & climate risk
         5) Save to memory and return aggregated report
        """
        log_event(f"orchestrator.start user={user_id}")

        # start or fetch session
        session = None
        try:
            session = self.session_service.start_session(user_id)
        except Exception as e:
            # defensive fallback: attempt get_session or create a minimal session object
            logger.warning("session_service.start_session failed: %s. Falling back to minimal session.", e)
            try:
                session = getattr(self.session_service, "get_session", lambda uid: None)(user_id)
            except Exception:
                session = None

            if session is None:
                # minimal ad-hoc session
                sess = type("SessionObj", (), {})()
                sess.user_id = user_id
                sess.context = {}
                sess.to_dict = lambda: {"user_id": user_id, "context": {}}
                session = sess

        user_memory = self.memory.get_user(user_id) or {}

        # 1) Diagnosis
        try:
            diag = self.diagnosis.run(image_input, crop_type=crop_type)
        except Exception as e:
            logger.exception("Diagnosis agent raised an exception: %s", e)
            diag = {"label": "error", "confidence": 0.0, "error": str(e)}

        log_event(f"diagnosis.result user={user_id} label={diag.get('label')} conf={diag.get('confidence')}")

        # 2) Advisory
        try:
            advice = self.advisory.run({"diagnosis": diag, "user_memory": user_memory, "crop": crop_type})
        except Exception as e:
            logger.exception("Advisory agent raised an exception: %s", e)
            advice = "Advisory generation failed: " + str(e)

        log_event(f"advisory.result user={user_id}")

        # 3) Market insights (best-effort)
        try:
            market_info = self.market.get_market_insights(crop_type or diag.get("crop") or "unknown")
        except Exception as e:
            logger.warning("Market agent failed: %s", e)
            market_info = {"error": str(e)}

        # 4) Climate risk (quick check)
        try:
            climate_info = self.climate.assess_location_risk(location=user_memory.get("location"))
        except Exception as e:
            logger.warning("Climate agent failed: %s", e)
            climate_info = {"error": str(e)}

        # 5) Persist to memory
        record = {
            "last_diag": diag,
            "last_advice": advice,
            "market": market_info,
            "climate": climate_info,
        }
        user_memory.update(record)
        try:
            self.memory.save_user(user_id, user_memory)
        except Exception as e:
            logger.warning("Saving to memory failed: %s", e)

        log_event(f"orchestrator.end user={user_id}")

        return {
            "diagnosis": diag,
            "advice": advice,
            "market": market_info,
            "climate": climate_info,
            "session": session.to_dict() if hasattr(session, "to_dict") else {},
        }


if __name__ == "__main__":
    print("Run the Orchestrator from a notebook. Example:")
    print("from agents.orchestrator import Orchestrator; Orchestrator().run_case('test', 'leaf1.png')")
