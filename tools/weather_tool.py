"""Location-aware short-range weather context using the public Open-Meteo APIs.

This returns general weather exposure signals, not a crop-specific disease forecast.
Thresholds are transparent heuristics and should be reviewed with local agronomists
before they are used for operational farming decisions.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
import logging

import requests

logger = logging.getLogger("agrisense.weather")

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherTool:
    def __init__(
        self,
        *,
        session: Optional[requests.Session] = None,
        timeout_seconds: float = 10.0,
    ):
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _parse_coordinates(location: str) -> Optional[Tuple[float, float]]:
        try:
            parts = [item.strip() for item in location.split(",")]
            if len(parts) != 2:
                return None
            lat, lon = float(parts[0]), float(parts[1])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        except (TypeError, ValueError):
            pass
        return None

    def _resolve_location(self, location: str) -> Dict[str, Any]:
        coordinates = self._parse_coordinates(location)
        if coordinates:
            return {
                "name": location,
                "latitude": coordinates[0],
                "longitude": coordinates[1],
                "country": None,
                "admin1": None,
            }

        response = self.session.get(
            GEOCODING_URL,
            params={"name": location, "count": 1, "language": "en", "format": "json"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        results = response.json().get("results") or []
        if not results:
            raise ValueError("Location not found. Enter a city name or latitude,longitude.")
        place = results[0]
        return {
            "name": place.get("name") or location,
            "latitude": float(place["latitude"]),
            "longitude": float(place["longitude"]),
            "country": place.get("country"),
            "admin1": place.get("admin1"),
        }

    @staticmethod
    def _summarize_risk(daily: Dict[str, Any]) -> Dict[str, Any]:
        """Apply explainable general weather-exposure thresholds to forecast days."""
        dates = daily.get("time") or []
        maxima = daily.get("temperature_2m_max") or []
        precipitation = daily.get("precipitation_sum") or []
        rain_probability = daily.get("precipitation_probability_max") or []
        wind = daily.get("wind_speed_10m_max") or []

        factors = []
        days = []
        for index, date in enumerate(dates):
            temp = maxima[index] if index < len(maxima) else None
            rain = precipitation[index] if index < len(precipitation) else None
            prob = rain_probability[index] if index < len(rain_probability) else None
            wind_speed = wind[index] if index < len(wind) else None

            day_factors = []
            if isinstance(temp, (int, float)) and temp >= 38:
                day_factors.append("high_heat")
            if isinstance(rain, (int, float)) and rain >= 30:
                day_factors.append("heavy_rainfall")
            if isinstance(prob, (int, float)) and prob >= 80:
                day_factors.append("high_rain_probability")
            if isinstance(wind_speed, (int, float)) and wind_speed >= 40:
                day_factors.append("strong_wind")

            factors.extend(day_factors)
            days.append({
                "date": date,
                "temperature_max_c": temp,
                "precipitation_sum_mm": rain,
                "precipitation_probability_max_percent": prob,
                "wind_speed_max_kmh": wind_speed,
                "risk_factors": day_factors,
            })

        unique_factors = sorted(set(factors))
        if any(factor in unique_factors for factor in ("high_heat", "heavy_rainfall", "strong_wind")):
            level = "high"
        elif unique_factors:
            level = "moderate"
        else:
            level = "low"

        return {"risk": level, "risk_factors": unique_factors, "daily_summary": days}

    def get_forecast_risk(self, location: Optional[str], days: int = 3) -> Dict[str, Any]:
        if not location or not str(location).strip():
            return {
                "location": location,
                "risk": "unknown",
                "note": "Enter a city, region, or latitude,longitude to enable weather context.",
            }

        try:
            resolved = self._resolve_location(str(location).strip())
            response = self.session.get(
                FORECAST_URL,
                params={
                    "latitude": resolved["latitude"],
                    "longitude": resolved["longitude"],
                    "daily": (
                        "temperature_2m_max,precipitation_sum,"
                        "precipitation_probability_max,wind_speed_10m_max"
                    ),
                    "forecast_days": max(1, min(int(days), 7)),
                    "timezone": "auto",
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            daily = payload.get("daily") or {}
            summary = self._summarize_risk(daily)
            if not summary["daily_summary"]:
                raise ValueError("Weather provider returned no daily forecast.")

            display_location = ", ".join(
                part for part in (resolved.get("name"), resolved.get("admin1"), resolved.get("country"))
                if part
            )
            return {
                "location": display_location or str(location),
                "latitude": resolved["latitude"],
                "longitude": resolved["longitude"],
                "timezone": payload.get("timezone"),
                "risk": summary["risk"],
                "risk_factors": summary["risk_factors"],
                "forecast": summary["daily_summary"],
                "source": "Open-Meteo Forecast API",
                "geocoding_source": "Open-Meteo Geocoding API" if not self._parse_coordinates(str(location)) else None,
                "fetched_at": self._now(),
                "forecast_window_days": len(summary["daily_summary"]),
                "note": (
                    "General weather-exposure heuristics for the selected forecast window; "
                    "not a crop-specific disease prediction."
                ),
                "method": {
                    "high_heat_threshold_c": 38,
                    "heavy_rainfall_threshold_mm_per_day": 30,
                    "high_rain_probability_threshold_percent": 80,
                    "strong_wind_threshold_kmh": 40,
                },
            }
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            logger.warning("Weather lookup failed for %s: %s", location, exc)
            return {
                "location": location,
                "risk": "unknown",
                "error": "weather_provider_unavailable",
                "note": str(exc),
                "source": "Open-Meteo Forecast API",
                "fetched_at": self._now(),
            }
