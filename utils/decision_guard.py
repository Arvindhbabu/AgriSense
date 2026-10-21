"""Uncertainty-aware decision gate for AgriSense.

This module does not improve a classifier's accuracy. It prevents low-confidence,
unknown, or demo-heuristic predictions from being treated as confirmed diagnoses.
Thresholds should be calibrated on a held-out, labeled dataset before deployment.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Mapping, Optional


def _confidence(value: Any) -> Optional[float]:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        return None
    return score


def is_demo_heuristic(diagnosis: Mapping[str, Any]) -> bool:
    """Detect the current RGB-statistics fallback used by DiagnosisAgent."""
    raw = diagnosis.get("raw")
    backend = str(diagnosis.get("backend", "")).strip().lower()
    return (
        backend == "demo_rgb_heuristic"
        or (
            isinstance(raw, Mapping)
            and {"r", "g", "b"}.issubset({str(key).lower() for key in raw.keys()})
            and not diagnosis.get("model_version")
        )
    )


def assess_diagnosis(
    diagnosis: Mapping[str, Any],
    min_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """Return a conservative, auditable decision about downstream advice.

    A passing result is only a *provisional candidate*, not a verified diagnosis.
    """
    threshold = min_confidence
    if threshold is None:
        try:
            threshold = float(os.getenv("AGRISENSE_MIN_CONFIDENCE", "0.70"))
        except ValueError:
            threshold = 0.70
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        threshold = 0.70
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        threshold = 0.70

    label = str(diagnosis.get("label", "unknown")).strip().lower()
    confidence = _confidence(diagnosis.get("confidence"))
    reasons = []

    if diagnosis.get("error") or label in {"", "unknown", "error", "none", "null"}:
        reasons.append("diagnosis_missing_or_backend_error")
    if is_demo_heuristic(diagnosis):
        reasons.append("demo_rgb_heuristic_is_not_disease_specific")
    if confidence is None:
        reasons.append("confidence_missing_or_invalid")
    elif confidence < threshold:
        reasons.append("reported_confidence_below_threshold")

    review_required = bool(reasons)
    return {
        "status": "review_required" if review_required else "provisional_candidate",
        "review_required": review_required,
        "allows_crop_specific_advice": not review_required,
        "reported_confidence": confidence,
        "minimum_confidence": threshold,
        "confidence_calibrated": bool(diagnosis.get("confidence_calibrated", False)),
        "reasons": reasons,
        "message": (
            "The result is uncertain or came from a demo heuristic. Capture a clearer image "
            "and seek local agricultural/extension review; do not choose a treatment from this result."
            if review_required
            else "This is a provisional model candidate. Confirm it against local agronomic guidance before treatment."
        ),
    }
