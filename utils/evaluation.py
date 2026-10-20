"""Metrics for evaluating crop diagnosis with uncertainty-aware abstention.

Input predictions are CSV rows with required columns y_true,y_pred and optional
confidence,review_required. A row marked review_required is treated as an abstention
for accepted-prediction metrics. Confidence is evaluated against correctness; it is
not assumed to be calibrated.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ABSTAIN_LABEL = "__ABSTAIN__"


def _is_true(value: Any) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes", "y"}


def load_prediction_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"y_true", "y_pred"}.issubset(set(reader.fieldnames)):
            raise ValueError("CSV must contain y_true and y_pred columns.")
        for line_number, row in enumerate(reader, start=2):
            truth = str(row.get("y_true") or "").strip()
            prediction = str(row.get("y_pred") or "").strip()
            if not truth or not prediction:
                raise ValueError("Missing y_true/y_pred value at CSV line " + str(line_number))
            confidence: Optional[float] = None
            raw_confidence = str(row.get("confidence") or "").strip()
            if raw_confidence:
                try:
                    confidence = float(raw_confidence)
                except ValueError as exc:
                    raise ValueError("Invalid confidence at CSV line " + str(line_number)) from exc
                if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                    raise ValueError("Confidence must be finite and within [0, 1] at CSV line " + str(line_number))
            rows.append({
                "y_true": truth,
                "y_pred": prediction,
                "confidence": confidence,
                "review_required": _is_true(row.get("review_required")),
            })
    if not rows:
        raise ValueError("CSV contains no prediction rows.")
    return rows


def _class_metrics(rows: List[Dict[str, Any]], classes: List[str]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    f1_values = []
    for label in classes:
        tp = sum(1 for row in rows if row["y_true"] == label and row["y_eval"] == label)
        fp = sum(1 for row in rows if row["y_true"] != label and row["y_eval"] == label)
        fn = sum(1 for row in rows if row["y_true"] == label and row["y_eval"] != label)
        support = sum(1 for row in rows if row["y_true"] == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        f1_values.append(f1)
    return {
        "per_class": metrics,
        "macro_precision": round(sum(item["precision"] for item in metrics.values()) / len(classes), 4) if classes else 0.0,
        "macro_recall": round(sum(item["recall"] for item in metrics.values()) / len(classes), 4) if classes else 0.0,
        "macro_f1": round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0,
    }


def _calibration_metrics(rows: List[Dict[str, Any]], bins: int) -> Dict[str, Any]:
    data = [
        (row["confidence"], 1.0 if row["y_pred"] == row["y_true"] else 0.0)
        for row in rows
        if row["confidence"] is not None and not row["review_required"]
    ]
    if not data:
        return {
            "sample_count": 0,
            "brier_score": None,
            "expected_calibration_error": None,
            "bins": [],
            "note": "No accepted predictions had confidence values.",
        }

    brier = sum((confidence - correct) ** 2 for confidence, correct in data) / len(data)
    ece = 0.0
    bin_reports = []
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = [
            (confidence, correct)
            for confidence, correct in data
            if lower <= confidence < upper or (bin_index == bins - 1 and confidence == 1.0)
        ]
        if not members:
            continue
        mean_confidence = sum(item[0] for item in members) / len(members)
        accuracy = sum(item[1] for item in members) / len(members)
        ece += (len(members) / len(data)) * abs(accuracy - mean_confidence)
        bin_reports.append({
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "count": len(members),
            "mean_confidence": round(mean_confidence, 4),
            "accuracy": round(accuracy, 4),
        })
    return {
        "sample_count": len(data),
        "brier_score": round(brier, 4),
        "expected_calibration_error": round(ece, 4),
        "bins": bin_reports,
        "note": "Confidence is evaluated against correctness; scores are not assumed calibrated.",
    }


def evaluate_predictions(rows: Iterable[Dict[str, Any]], bins: int = 10) -> Dict[str, Any]:
    if not isinstance(bins, int) or bins < 2 or bins > 100:
        raise ValueError("bins must be an integer from 2 through 100.")
    source_rows = list(rows)
    if not source_rows:
        raise ValueError("At least one prediction row is required.")

    normalized = []
    for row in source_rows:
        truth = str(row.get("y_true", "")).strip()
        prediction = str(row.get("y_pred", "")).strip()
        if not truth or not prediction:
            raise ValueError("Every row must provide non-empty y_true and y_pred values.")
        if truth == ABSTAIN_LABEL or prediction == ABSTAIN_LABEL:
            raise ValueError(ABSTAIN_LABEL + " is reserved for abstentions.")
        confidence = row.get("confidence")
        if confidence not in (None, ""):
            try:
                confidence = float(confidence)
            except (TypeError, ValueError) as exc:
                raise ValueError("confidence must be numeric when supplied.") from exc
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be finite and within [0, 1].")
        else:
            confidence = None
        review_required = _is_true(row.get("review_required"))
        normalized.append({
            "y_true": truth,
            "y_pred": prediction,
            "confidence": confidence,
            "review_required": review_required,
            "y_eval": ABSTAIN_LABEL if review_required else prediction,
        })

    classes = sorted(
        set(row["y_true"] for row in normalized)
        | set(row["y_pred"] for row in normalized if not row["review_required"])
    )
    evaluation_labels = classes + [ABSTAIN_LABEL]
    confusion = {
        truth: {
            predicted: sum(1 for row in normalized if row["y_true"] == truth and row["y_eval"] == predicted)
            for predicted in evaluation_labels
        }
        for truth in classes
    }
    accepted = [row for row in normalized if not row["review_required"]]
    correct_accepted = sum(1 for row in accepted if row["y_pred"] == row["y_true"])
    coverage = len(accepted) / len(normalized)
    selective_accuracy = correct_accepted / len(accepted) if accepted else None

    class_metrics = _class_metrics(normalized, classes)
    return {
        "sample_count": len(normalized),
        "class_labels": classes,
        "abstain_label": ABSTAIN_LABEL,
        "classification": class_metrics,
        "confusion_matrix": confusion,
        "abstention": {
            "accepted_count": len(accepted),
            "abstained_count": len(normalized) - len(accepted),
            "coverage": round(coverage, 4),
            "selective_accuracy": round(selective_accuracy, 4) if selective_accuracy is not None else None,
            "selective_risk": round(1.0 - selective_accuracy, 4) if selective_accuracy is not None else None,
            "abstention_rate": round(1.0 - coverage, 4),
        },
        "calibration": _calibration_metrics(normalized, bins),
        "methodology": {
            "review_required_rows_are_abstentions": True,
            "confidence_assessed_as_probability_of_correct_label": True,
            "calibration_is_not_assumed": True,
            "recommended_use": "Compare baseline and uncertainty-gated systems on the same held-out labelled examples.",
        },
    }


def evaluate_csv(input_path: str, output_path: str, bins: int = 10) -> Dict[str, Any]:
    report = evaluate_predictions(load_prediction_csv(input_path), bins=bins)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return report
