# agents/market_agent.py
"""
MarketAgent: fetches market prices, simple demand signals.
This is intentionally simple — for production use, wire to local market APIs.
"""
import logging
from typing import Optional

logger = logging.getLogger("agrisense.market")
logger.setLevel(logging.INFO)


class MarketAgent:
    def __init__(self, market_tool=None):
        """
        market_tool: an object exposing .get_price(crop, region) or similar.
        If None, the agent will return placeholder values.
        """
        self.market_tool = market_tool

    def get_market_insights(self, crop: str, region: Optional[str] = None):
        if self.market_tool:
            return self.market_tool.get_price(crop, region)
        # Placeholder logic: returns a dummy structure
        logger.info("MarketAgent: no market_tool, returning placeholder.")
        return {
            "crop": crop,
            "region": region or "unknown",
            "avg_price_per_kg": None,
            "trend": "unknown",
            "note": "No market_tool provided. Connect an OpenAPI for live prices."
        }
