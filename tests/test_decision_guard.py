import unittest

from utils.decision_guard import assess_diagnosis, is_demo_heuristic


class DecisionGuardTests(unittest.TestCase):
    def test_demo_rgb_heuristic_must_be_reviewed_even_with_high_score(self):
        diagnosis = {
            "label": "healthy",
            "confidence": 0.99,
            "raw": {"r": 0.2, "g": 0.8, "b": 0.1},
        }
        self.assertTrue(is_demo_heuristic(diagnosis))
        decision = assess_diagnosis(diagnosis)
        self.assertTrue(decision["review_required"])
        self.assertFalse(decision["allows_crop_specific_advice"])
        self.assertIn("demo_rgb_heuristic_is_not_disease_specific", decision["reasons"])

    def test_unknown_prediction_requires_review(self):
        decision = assess_diagnosis({"label": "unknown", "confidence": 0.0})
        self.assertEqual(decision["status"], "review_required")

    def test_low_confidence_requires_review(self):
        decision = assess_diagnosis({"label": "leaf_spot", "confidence": 0.55}, min_confidence=0.7)
        self.assertTrue(decision["review_required"])
        self.assertIn("reported_confidence_below_threshold", decision["reasons"])

    def test_high_confidence_non_heuristic_is_only_provisional(self):
        decision = assess_diagnosis({"label": "leaf_spot", "confidence": 0.91}, min_confidence=0.7)
        self.assertEqual(decision["status"], "provisional_candidate")
        self.assertTrue(decision["allows_crop_specific_advice"])
        self.assertFalse(decision["confidence_calibrated"])

    def test_invalid_confidence_requires_review(self):
        decision = assess_diagnosis({"label": "leaf_spot", "confidence": float("nan")})
        self.assertTrue(decision["review_required"])
        self.assertIn("confidence_missing_or_invalid", decision["reasons"])


if __name__ == "__main__":
    unittest.main()
