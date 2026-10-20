import csv
import json
import os
import tempfile
import unittest

from utils.evaluation import ABSTAIN_LABEL, evaluate_csv, evaluate_predictions, load_prediction_csv


class EvaluationTests(unittest.TestCase):
    def test_abstention_coverage_risk_and_calibration_are_reported(self):
        rows = [
            {"y_true": "A", "y_pred": "A", "confidence": 0.9, "review_required": False},
            {"y_true": "A", "y_pred": "B", "confidence": 0.6, "review_required": False},
            {"y_true": "B", "y_pred": "B", "confidence": 0.9, "review_required": False},
            {"y_true": "B", "y_pred": "A", "confidence": 0.4, "review_required": True},
        ]
        report = evaluate_predictions(rows, bins=5)
        self.assertEqual(report["sample_count"], 4)
        self.assertEqual(report["abstention"]["accepted_count"], 3)
        self.assertEqual(report["abstention"]["abstained_count"], 1)
        self.assertEqual(report["abstention"]["coverage"], 0.75)
        self.assertEqual(report["abstention"]["selective_accuracy"], 0.6667)
        self.assertEqual(report["abstention"]["selective_risk"], 0.3333)
        self.assertIsNotNone(report["calibration"]["brier_score"])
        self.assertIsNotNone(report["calibration"]["expected_calibration_error"])
        self.assertEqual(report["confusion_matrix"]["B"][ABSTAIN_LABEL], 1)

    def test_prediction_only_class_is_in_macro_metrics(self):
        report = evaluate_predictions([
            {"y_true": "A", "y_pred": "A", "confidence": 0.9},
            {"y_true": "A", "y_pred": "C", "confidence": 0.2},
        ])
        self.assertIn("C", report["classification"]["per_class"])
        self.assertEqual(report["classification"]["per_class"]["C"]["support"], 0)
        self.assertEqual(report["classification"]["per_class"]["C"]["f1"], 0.0)

    def test_empty_rows_and_invalid_confidence_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_predictions([])
        with self.assertRaises(ValueError):
            evaluate_predictions([{"y_true": "A", "y_pred": "A", "confidence": 1.2}])

    def test_csv_load_and_report_write(self):
        with tempfile.TemporaryDirectory() as folder:
            csv_path = os.path.join(folder, "predictions.csv")
            json_path = os.path.join(folder, "metrics.json")
            with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["y_true", "y_pred", "confidence", "review_required"],
                )
                writer.writeheader()
                writer.writerow({"y_true": "healthy", "y_pred": "healthy", "confidence": "0.9", "review_required": "false"})
                writer.writerow({"y_true": "blight", "y_pred": "unknown", "confidence": "0.4", "review_required": "true"})

            rows = load_prediction_csv(csv_path)
            self.assertEqual(len(rows), 2)
            report = evaluate_csv(csv_path, json_path)
            self.assertTrue(os.path.isfile(json_path))
            with open(json_path, "r", encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertEqual(saved["sample_count"], 2)
            self.assertEqual(report["abstention"]["coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()
