import unittest

from tools.market_tool import MarketTool


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        return FakeResponse(self.payload)


class MarketToolTests(unittest.TestCase):
    def test_unconfigured_market_is_explicitly_unavailable(self):
        result = MarketTool(api_key="", resource_id="").get_price("wheat")
        self.assertIsNone(result["avg_price_per_kg"])
        self.assertEqual(result["trend"], "unknown")
        self.assertIn("DATA_GOV_IN_API_KEY", result["note"])

    def test_agmarknet_modal_price_is_converted_from_quintal_to_kg(self):
        session = FakeSession({
            "records": [{
                "commodity": "Wheat",
                "state": "Tamil Nadu",
                "market": "Test Mandi",
                "arrival_date": "20/10/2026",
                "modal_price": "3000",
                "min_price": "2800",
                "max_price": "3200",
            }]
        })
        tool = MarketTool(
            api_key="test-key",
            resource_id="test-resource",
            session=session,
        )
        result = tool.get_price("Wheat", "Tamil Nadu")

        self.assertEqual(result["avg_price_per_kg"], 30.0)
        self.assertEqual(result["market"], "Test Mandi")
        self.assertEqual(result["source"], "data.gov.in Agmarknet resource")
        self.assertEqual(result["trend"], "unknown")
        self.assertIn("filters[state]", session.calls[0][1])

    def test_empty_dataset_does_not_invent_a_price(self):
        tool = MarketTool(
            api_key="test-key",
            resource_id="test-resource",
            session=FakeSession({"records": []}),
        )
        result = tool.get_price("Rice")
        self.assertIsNone(result["avg_price_per_kg"])
        self.assertIn("No matching records", result["note"])


if __name__ == "__main__":
    unittest.main()
