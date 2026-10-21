"""Lightweight image-quality checks before crop-image triage.

These are image-acquisition checks, not a plant/disease detector. Thresholds are
transparent heuristics and should be evaluated with the target device and dataset.
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np
from PIL import Image


def assess_image_quality(image: Image.Image) -> Dict[str, Any]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    thumbnail = rgb.resize((128, 128))
    gray = np.asarray(thumbnail.convert("L"), dtype=np.float32) / 255.0
    brightness = float(np.mean(gray))
    contrast_std = float(np.std(gray))

    warnings = []
    if min(width, height) < 96:
        warnings.append("very_low_resolution")
    if brightness < 0.12:
        warnings.append("severely_underexposed")
    elif brightness > 0.94:
        warnings.append("severely_overexposed")
    if contrast_std < 0.025:
        warnings.append("very_low_contrast_or_nearly_uniform_image")

    # A transparent heuristic score for triage only; not a learned/calibrated probability.
    score = 1.0
    if min(width, height) < 256:
        score -= 0.15
    if brightness < 0.20 or brightness > 0.88:
        score -= 0.20
    if contrast_std < 0.06:
        score -= 0.20
    score = round(max(0.0, min(1.0, score)), 3)

    return {
        "width": int(width),
        "height": int(height),
        "mean_brightness": round(brightness, 4),
        "contrast_std": round(contrast_std, 4),
        "quality_score": score,
        "score_is_calibrated": False,
        "warnings": warnings,
        "review_required": bool(warnings),
        "method": "pillow_grayscale_brightness_and_contrast_heuristics",
    }
