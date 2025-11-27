# agents/climate_agent.py
"""
ClimateAgent: short-term climate risk assessment.
Optionally runs as a long-running background watcher (simple thread-based).
"""
import logging
import threading
import time
from typing import Optional, Dict, Any

logger = logging.getLogger("agrisense.climate")
logger.setLevel(logging.INFO)


class ClimateAgent:
    def __init__(self, weather_tool=None):
        self.weather_tool = weather_tool
        self._watch_thread = None
        self._running = False

    def assess_location_risk(self, location: Optional[str]) -> Dict[str, Any]:
        # If weather_tool provided, call it; else placeholder
        if self.weather_tool and location:
            return self.weather_tool.get_forecast_risk(location)
        return {"location": location, "risk": "unknown", "note": "No weather_tool connected."}

    def start_background_monitor(self, location_list, interval_seconds=6 * 3600):
        if self._running:
            logger.info("ClimateAgent monitor already running.")
            return
        self._running = True

        def runner():
            while self._running:
                for loc in location_list:
                    try:
                        info = self.assess_location_risk(loc)
                        logger.info("Climate watch %s -> %s", loc, info)
                    except Exception as e:
                        logger.warning("Climate monitor error for %s: %s", loc, e)
                time.sleep(interval_seconds)

        self._watch_thread = threading.Thread(target=runner, daemon=True)
        self._watch_thread.start()

    def stop_background_monitor(self):
        self._running = False
        if self._watch_thread:
            self._watch_thread.join(timeout=2)
