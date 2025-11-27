# agents/advisory_agent.py
"""
AdvisoryAgent: generates human-friendly, localized advice from a diagnosis.
It prefers Gemini if provided; otherwise falls back to a template-based responder.
"""
import logging
from typing import Dict, Any

logger = logging.getLogger("agrisense.advisory")
logger.setLevel(logging.INFO)


class AdvisoryAgent:
    def __init__(self, gemini_client=None, language="en"):
        self.gemini = gemini_client
        self.language = language

    def _prompt_for_gemini(self, diagnosis: Dict[str, Any], user_memory: Dict[str, Any], crop: str):
        label = diagnosis.get("label", "unknown")
        conf = diagnosis.get("confidence", 0.0)
        context = user_memory or {}
        prompt = (
            f"You are an expert agricultural advisor. The farmer has a crop: {crop}.\n"
            f"Diagnosis: {label} (confidence: {conf:.2f}).\n"
            f"Provide concise actionable steps: immediate actions, treatment (name & dosage), safety precautions, "
            f"organic alternatives if available, and follow-up timing. Keep instructions clear for a smallholder farmer."
        )
        return prompt

    def _call_gemini(self, prompt):
        # PLACEHOLDER: adapt to exact gemini client you've used in notebooks
        # Example pseudocode:
        # from google.genai import types
        # response = self.gemini.text_generation.create(model="gemini-2.5", prompt=prompt, temperature=0.2)
        # return response.text
        raise NotImplementedError("Please implement _call_gemini with your gemini client snippet from notebooks.")

    def _rule_based_advice(self, diagnosis):
        label = diagnosis.get("label", "unknown")
        # Very minimal mapping — extend with more clinical rules or template library
        mapping = {
            "leaf_blight": (
                "1) Remove heavily affected leaves and burn/biodeposit them away from field.\n"
                "2) Apply a recommended fungicide (e.g., Azoxystrobin) following label dose — *replace with local product & dose*.\n"
                "3) Avoid spraying during peak pollinator activity. Re-check after 7 days."
            ),
            "leaf_spot": (
                "1) Remove infected leaves.\n2) Improve ventilation and reduce humidity.\n3) Use copper-based sprays as per label instructions."
            ),
            "healthy": "No action required. Continue monitoring weekly.",
            "unknown": "Please collect a clearer image and re-upload; or consult a local extension officer."
        }
        return mapping.get(label, mapping["unknown"])

    def run(self, payload: Dict[str, Any]) -> str:
        diagnosis = payload.get("diagnosis", {})
        user_memory = payload.get("user_memory", {})
        crop = payload.get("crop", "unknown")

        # Try Gemini first
        if self.gemini:
            try:
                prompt = self._prompt_for_gemini(diagnosis, user_memory, crop)
                return self._call_gemini(prompt)
            except Exception as e:
                logger.warning("Gemini advisory failed: %s", e)

        # Fallback: rule/template based
        return self._rule_based_advice(diagnosis)
