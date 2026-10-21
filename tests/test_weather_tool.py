import unittest

from tools.weather_tool import WeatherTool, GEOCODING_URL, FORECAST_URL


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        if url == GEOCODING_URL:
            return FakeResponse({
                "results": [{
                    "name": "Demo City",
                    "latitude": 13.08,
                    "longitude": 80.27,
                    "country": "India",
                    "admin1": "Tamil Nadu",
                }]
            })
        if url == FORECAST_URL:
            return FakeResponse({
                "timezone": "Asia/Kolkata",
                "daily": {
                    "time": ["2026-10-20", "2026-10-21"],
                    "temperature_2m_max": [39.0, 33.0],
                    "precipitation_sum": [2.0, 4.0],
                    "precipitation_probability_max": [10, 20],
                    "wind_speed_10m_max": [12.0, 15.0],
                },
            })
        raise AssertionError("Unexpected URL: " + url)


class WeatherToolTests(unittest.TestCase):
    def test_no_location_returns_unknown_without_network(self):
        tool = WeatherTool(session=FakeSession())
        result = tool.get_forecast_risk("")
        self.assertEqual(result["risk"], "unknown")
        self.assertIn("Enter a city", result["note"])

    def test_forecast_is_source_attributed_and_flags_heat(self):
        session = FakeSession()
        result = WeatherTool(session=session).get_forecast_risk("Demo City")

        self.assertEqual(result["risk"], "high")
        self.assertIn("high_heat", result["risk_factors"])
        self.assertEqual(result["source"], "Open-Meteo Forecast API")
        self.assertIn("fetched_at", result)
        self.assertNotIn("observed_at", result)
        self.assertEqual(len(result["forecast"]), 2)
        self.assertEqual(session.calls[0][0], GEOCODING_URL)
        self.assertEqual(session.calls[1][0], FORECAST_URL)

    def test_coordinates_skip_geocoding(self):
        session = FakeSession()
        # The fake session's first call would normally return a geocoding response;
        # prefill a forecast-only session so coordinate lookup uses the forecast endpoint.
        class ForecastOnlySession:
            def get(self, url, params=None, timeout=None):
                self_url = url
                if self_url != FORECAST_URL:
                    raise AssertionError("Coordinates should skip geocoding")
                return FakeResponse({
                    "timezone": "UTC",
                    "daily": {
                        "time": ["2026-10-20"],
                        "temperature_2m_max": [30.0],
                        "precipitation_sum": [0.0],
                        "precipitation_probability_max": [0],
                        "wind_speed_10m_max": [10.0],
                    },
                })

        result = WeatherTool(session=ForecastOnlySession()).get_forecast_risk("13.08,80.27")
        self.assertEqual(result["risk"], "low")
        self.assertAlmostEqual(result["latitude"], 13.08)
        self.assertIsNone(result["geocoding_source"])


if __name__ == "__main__":
    unittest.main()
