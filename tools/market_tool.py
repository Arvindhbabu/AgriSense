"""Configurable Indian mandi-price adapter using data.gov.in's resource API.

Configure DATA_GOV_IN_API_KEY and DATA_GOV_IN_MARKET_RESOURCE_ID from the official
Open Government Data portal. Resource schemas vary; this adapter tolerates common
Agmarknet field names and reports unavailable rather than fabricating prices.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional
import logging
import os

import requests

logger = logging.getLogger("agrisense.market")


class MarketTool:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        resource_id: Optional[str] = None,
        session: Optional[requests.Session] = None,
        timeout_seconds: float = 12.0,
    ):
        self.api_key = api_key or os.getenv("DATA_GOV_IN_API_KEY")
        self.resource_id = resource_id or os.getenv("DATA_GOV_IN_MARKET_RESOURCE_ID")
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds
        self.price_unit = os.getenv("DATA_GOV_IN_PRICE_UNIT", "quintal").strip().lower()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _first(record: Dict[str, Any], *keys: str):
        lowered = {str(key).strip().lower().replace(" ", "_"): value for key, value in record.items()}
        for key in keys:
            value = lowered.get(key.lower().replace(" ", "_"))
            if value not in (None, ""):
                return value
        return None

    @staticmethod
    def _number(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            parsed = float(str(value).replace(",", "").replace("₹", "").strip())
            return parsed if parsed >= 0 else None
        except (TypeError, ValueError):
            return None

    def get_price(self, crop: str, region: Optional[str] = None) -> Dict[str, Any]:
        if not self.api_key or not self.resource_id:
            return {
                "crop": crop,
                "region": region or "India",
                "avg_price_per_kg": None,
                "trend": "unknown",
                "note": (
                    "Market data is not configured. Set DATA_GOV_IN_API_KEY and "
                    "DATA_GOV_IN_MARKET_RESOURCE_ID using the official data.gov.in portal."
                ),
            }

        url = "https://api.data.gov.in/resource/" + self.resource_id
        params = {"api-key": self.api_key, "format": "json", "limit": 100, "offset": 0}
        params["filters[commodity]"] = str(crop).strip().title()
        if region:
            params["filters[state]"] = region

        try:
            response = self.session.get(url, params=params, timeout=self.timeout_seconds)
            response.raise_for_status()
            records = response.json().get("records") or []
            if not records:
                return {
                    "crop": crop,
                    "region": region or "India",
                    "avg_price_per_kg": None,
                    "trend": "unknown",
                    "source": "data.gov.in Agmarknet resource",
                    "fetched_at": self._now(),
                    "note": "No matching records returned; verify the configured resource filters and schema.",
                }

            def rank(record):
                return str(self._first(record, "arrival_date", "date", "reported_date") or "")

            records.sort(key=rank, reverse=True)
            record = records[0]
            commodity = self._first(record, "commodity", "crop") or crop
            state = self._first(record, "state") or region
            market = self._first(record, "market", "mandi", "market_name")
            arrival_date = self._first(record, "arrival_date", "date", "reported_date")
            modal = self._number(self._first(record, "modal_price", "modal price", "price"))
            min_price = self._number(self._first(record, "min_price", "minimum_price", "min price"))
            max_price = self._number(self._first(record, "max_price", "maximum_price", "max price"))

            price_per_kg = None
            if modal is not None:
                if self.price_unit in {"quintal", "quintals", "100kg"}:
                    price_per_kg = round(modal / 100.0, 4)
                elif self.price_unit in {"kg", "kilogram", "kilograms"}:
                    price_per_kg = round(modal, 4)

            return {
                "crop": str(commodity),
                "region": str(state or "India"),
                "market": str(market) if market else None,
                "avg_price_per_kg": price_per_kg,
                "modal_price": modal,
                "min_price": min_price,
                "max_price": max_price,
                "price_unit": "INR/kg" if price_per_kg is not None else self.price_unit,
                "trend": "unknown",
                "source": "data.gov.in Agmarknet resource",
                "observed_at": str(arrival_date) if arrival_date else None,
                "fetched_at": self._now(),
                "record_count": len(records),
                "note": (
                    "Most recent matching modal price from the configured dataset. A trend is not inferred "
                    "from one record; verify commodity, market, date, and unit before deciding."
                ),
            }
        except (requests.RequestException, ValueError, TypeError) as exc:
            # Do not log the exception string: HTTP errors can include the API key in the query URL.
            logger.warning("Market API lookup failed for crop=%s region=%s (%s)", crop, region, type(exc).__name__)
            return {
                "crop": crop,
                "region": region or "India",
                "avg_price_per_kg": None,
                "trend": "unknown",
                "source": "data.gov.in Agmarknet resource",
                "fetched_at": self._now(),
                "error": "market_provider_unavailable",
                "note": "Market request failed; check the API key, resource ID, network, and dataset schema.",
            }
