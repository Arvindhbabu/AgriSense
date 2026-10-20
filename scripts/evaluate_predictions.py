"""Evaluate a labelled crop-prediction CSV.

Example:
  python scripts/evaluate_predictions.py --input predictions.csv --output reports/evaluation.json

Required columns: y_true,y_pred
Optional columns: confidence,review_required
"""
from __future__ import annotations

import argparse
import json
import sys

from utils.evaluation import evaluate_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate crop diagnosis and abstention metrics.")
    parser.add_argument("--input", required=True, help="CSV with y_true,y_pred and optional confidence,review_required")
    parser.add_argument("--output", default="reports/evaluation.json", help="Destination JSON report")
    parser.add_argument("--bins", type=int, default=10, help="Expected calibration error bins (2-100)")
    args = parser.parse_args()

    try:
        report = evaluate_csv(args.input, args.output, bins=args.bins)
    except (OSError, ValueError) as exc:
        print("Evaluation failed: " + str(exc), file=sys.stderr)
        return 2

    print(json.dumps({
        "samples": report["sample_count"],
        "macro_f1": report["classification"]["macro_f1"],
        "coverage": report["abstention"]["coverage"],
        "selective_accuracy": report["abstention"]["selective_accuracy"],
        "selective_risk": report["abstention"]["selective_risk"],
        "brier_score": report["calibration"]["brier_score"],
        "expected_calibration_error": report["calibration"]["expected_calibration_error"],
        "report": args.output,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
