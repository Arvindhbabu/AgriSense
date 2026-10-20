import unittest

from agents.orchestrator import Orchestrator


class FakeDiagnosis:
    def __init__(self, value):
        self.value = value

    def run(self, image_input, crop_type=None):
        return dict(self.value)


class FakeAdvisory:
    def __init__(self):
        self.called = False

    def run(self, payload):
        self.called = True
        return "Provisional recommendation; validate with local guidance."


class FakeMarket:
    def get_market_insights(self, crop, region=None):
        return {
            "crop": crop,
            "region": region or "demo-region",
            "avg_price_per_kg": 28.5,
            "trend": "stable",
            "source": "test-market-adapter",
            "observed_at": "2026-10-19T10:00:00+00:00",
        }


class FakeClimate:
    def assess_location_risk(self, location=None):
        return {
            "location": location or "demo-location",
            "risk": "moderate",
            "source": "test-weather-adapter",
            "observed_at": "2026-10-19T09:00:00+00:00",
        }


class OrchestratorTests(unittest.TestCase):
    def build_orchestrator(self, diagnosis):
        instance = Orchestrator(min_diagnosis_confidence=0.7)
        instance.diagnosis = FakeDiagnosis(diagnosis)
        instance.advisory = FakeAdvisory()
        instance.market = FakeMarket()
        instance.climate = FakeClimate()
        return instance

    def test_demo_heuristic_suppresses_crop_specific_advice(self):
        instance = self.build_orchestrator({
            "label": "healthy",
            "confidence": 0.99,
            "raw": {"r": 0.2, "g": 0.8, "b": 0.1},
        })
        events = []
        report = instance.run_case("test-farmer", object(), "wheat", lambda step, info=None: events.append(step))

        self.assertEqual(report["diagnosis_gate"]["status"], "review_required")
        self.assertEqual(report["advisory_mode"], "uncertainty_guard")
        self.assertFalse(instance.advisory.called)
        self.assertIn("case.completed", events)
        self.assertIn("advisory.skipped", events)
        self.assertEqual(report["market"]["evidence"]["status"], "available_with_provenance")
        self.assertEqual(report["climate"]["evidence"]["status"], "available_with_provenance")

    def test_confident_non_heuristic_is_still_marked_provisional(self):
        instance = self.build_orchestrator({
            "label": "leaf_spot",
            "confidence": 0.91,
            "model_version": "test-model-1",
            "confidence_calibrated": False,
        })
        report = instance.run_case("test-farmer", object(), "tomato")

        self.assertEqual(report["diagnosis_gate"]["status"], "provisional_candidate")
        self.assertTrue(instance.advisory.called)
        self.assertIn("not a confirmed diagnosis", report["safety_note"].lower())
        self.assertEqual(report["run_metadata"]["pipeline"], "agrisense-evidence-gated-v1")


if __name__ == "__main__":
    unittest.main()
