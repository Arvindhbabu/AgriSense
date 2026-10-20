"""Conservative agricultural advisory agent.

Gemini is optional. The template fallback is deliberately non-prescriptive and does
not invent pesticide rates. Crop- and locality-specific treatment must be verified
against reliable local extension guidance and official product labels.
"""
import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger("agrisense.advisory")
logger.setLevel(logging.INFO)


class AdvisoryAgent:
    def __init__(self, gemini_client=None, language="en"):
        self.gemini = gemini_client
        self.language = language

    def _prompt_for_gemini(self, diagnosis: Dict[str, Any], user_memory: Dict[str, Any], crop: str):
        # Keep prompt context compact; raw model tensors/images should not enter advisory context.
        market = user_memory.get("_current_market_evidence", {})
        climate = user_memory.get("_current_climate_evidence", {})
        compact_diagnosis = {
            key: diagnosis.get(key)
            for key in ("label", "confidence", "notes", "needs_expert_review", "backend", "model_version")
            if diagnosis.get(key) is not None
        }
        context = {
            "crop": crop or "unspecified",
            "location": user_memory.get("location"),
            "diagnosis": compact_diagnosis,
            "market_evidence": market,
            "climate_evidence": climate,
            "language": self.language,
        }
        return (
            "You are an agricultural decision-support assistant. Use only the facts in this JSON context; "
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
                "1. Inspect several plants and photograph both the affected and healthy leaves.\n"
                "2. Avoid working through wet foliage; clean tools before moving between affected areas.\n"
                "3. Remove badly affected material only where appropriate for this crop and local guidance.\n"
                "4. Confirm the disease with a local agricultural extension officer before selecting a treatment."
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
        crop = payload.get("crop", "unknown")

        # Inject only the evidence summary for this case; do not confuse it with persistent farmer memory.
        user_memory["_current_market_evidence"] = payload.get("market", {})
        user_memory["_current_climate_evidence"] = payload.get("climate", {})

        if self.gemini:
            try:
                prompt = self._prompt_for_gemini(diagnosis, user_memory, crop)
                return self._call_gemini(prompt)
            except Exception as exc:
                logger.warning("Gemini advisory failed; using conservative fallback: %s", exc)

        return self._rule_based_advice(diagnosis)
