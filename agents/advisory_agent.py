"""Conservative agricultural advisory agent.

The agent consumes diagnosis plus climate/market evidence. It deliberately avoids
inventing prices, weather, or chemical rates. Gemini is optional.
"""
import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger("agrisense.advisory")
logger.setLevel(logging.INFO)


def _compact_evidence(value: Any) -> Dict[str, Any]:
    """Pass only decision-relevant fields to the LLM, not full raw forecast payloads."""
    if not isinstance(value, dict):
        return {"status": "unavailable"}
    evidence = value.get("evidence", {})
    result = {
        "status": evidence.get("status", "unverified") if isinstance(evidence, dict) else "unverified",
    }
    for key in (
        "crop", "region", "avg_price_per_kg", "trend", "risk",
        "location", "note", "source", "observed_at", "fetched_at", "timestamp",
    ):
        candidate = value.get(key)
        if candidate is not None and isinstance(candidate, (str, int, float, bool)):
            result[key] = candidate
    if isinstance(evidence, dict):
        for key in ("source", "observed_at", "freshness"):
            if evidence.get(key) is not None:
                result["evidence_" + key] = evidence.get(key)
    return result


class AdvisoryAgent:
    def __init__(self, gemini_client=None, language="en"):
        self.gemini = gemini_client
        self.language = language

    def _prompt_for_gemini(self, diagnosis: Dict[str, Any], user_memory: Dict[str, Any], crop: str):
        compact_diagnosis = {
            key: diagnosis.get(key)
            for key in (
                "label", "confidence", "notes", "needs_expert_review",
                "backend", "model_version", "confidence_calibrated",
            )
            if diagnosis.get(key) is not None
        }
        context = {
            "crop": crop or "unspecified",
            "location": user_memory.get("location"),
            "diagnosis": compact_diagnosis,
            "market_evidence": _compact_evidence(user_memory.get("_current_market_evidence", {})),
            "climate_evidence": _compact_evidence(user_memory.get("_current_climate_evidence", {})),
            "language": self.language,
        }
        return (
            "You are an agricultural decision-support assistant. Use only facts in this JSON context; "
            "do not invent disease evidence, weather, prices, chemical products, or dosages. The diagnosis "
            "is provisional, not ground truth. If diagnosis is unclear or needs expert review, recommend "
            "re-taking a clear image and contacting a local agricultural extension officer, and do not give "
            "crop-specific pesticide advice. Prefer low-risk cultural/preventive steps and explain what to "
            "monitor. Never give pesticide dosage or tank-mix instructions; direct the farmer to approved "
            "local product labels and qualified local guidance. Explicitly say when market/climate evidence "
            "is unavailable. Keep advice practical, concise, and in the requested language.\n\n"
            "CONTEXT JSON:\n" + json.dumps(context, ensure_ascii=False, default=str)
        )

    def _call_gemini(self, prompt):
        if self.gemini is None:
            raise RuntimeError("Gemini client is not configured.")
        models = getattr(self.gemini, "models", None)
        generate = getattr(models, "generate_content", None)
        if not callable(generate):
            raise RuntimeError("Gemini client has no models.generate_content method.")
        model_name = (
            os.getenv("GEMINI_ADVISORY_MODEL")
            or os.getenv("GEMINI_MODEL")
            or "gemini-3.8-flash"
        )
        response = generate(model=model_name, contents=prompt)
        text = getattr(response, "text", None)
        if not text or not str(text).strip():
            raise RuntimeError("Gemini advisory response was empty.")
        return str(text).strip()

    def _rule_based_advice(self, diagnosis):
        label = str(diagnosis.get("label", "unknown")).strip().lower()
        mapping = {
            "leaf_blight": (
                "Provisional next steps:\n"
                "1. Inspect several plants and photograph both affected and healthy leaves.\n"
                "2. Avoid working through wet foliage; clean tools before moving between affected areas.\n"
                "3. Remove badly affected material only where appropriate for this crop and local guidance.\n"
                "4. Confirm the disease locally before selecting a treatment."
            ),
            "leaf_spot": (
                "Provisional next steps:\n"
                "1. Photograph the spots in good light and check whether symptoms are spreading.\n"
                "2. Avoid unnecessary overhead watering and reduce prolonged leaf wetness where practical.\n"
                "3. Compare symptoms across multiple plants and confirm the cause locally before treatment."
            ),
            "healthy": (
                "No clear issue was identified by the configured backend, but this is not proof that the crop "
                "is disease-free. Continue routine scouting and re-check if symptoms appear or spread."
            ),
            "unknown": (
                "The cause could not be identified reliably. Take a sharp image in natural light showing the "
                "whole leaf and close-up symptoms, note the crop and recent field conditions, and consult a "
                "local agricultural extension officer."
            ),
            "error": (
                "The diagnosis service could not provide a reliable result. Try a clearer crop image or consult "
                "a local agricultural extension officer."
            ),
        }
        return mapping.get(label, mapping["unknown"])

    def run(self, payload: Dict[str, Any]) -> str:
        diagnosis = payload.get("diagnosis", {}) or {}
        user_memory = dict(payload.get("user_memory", {}) or {})
        user_memory["_current_market_evidence"] = payload.get("market", {})
        user_memory["_current_climate_evidence"] = payload.get("climate", {})
        crop = payload.get("crop", "unknown")

        if self.gemini:
            try:
                prompt = self._prompt_for_gemini(diagnosis, user_memory, crop)
                return self._call_gemini(prompt)
            except Exception as exc:
                logger.warning("Gemini advisory failed; using conservative fallback: %s", exc)

        return self._rule_based_advice(diagnosis)
