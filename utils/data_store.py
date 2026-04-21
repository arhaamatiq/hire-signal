import json
import os
import re
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from utils.config_loader import load_config

logger = logging.getLogger(__name__)


class DataStore:
    """
    Lightweight JSON-based persistence layer for company snapshots.

    Each company gets its own directory under history_dir/:
        data/historical/anthropic/2025-03-01.json
        data/historical/anthropic/2025-03-08.json
        ...

    This enables week-over-week delta computation across nightly CI runs.
    """

    def __init__(self, history_dir: Optional[str] = None):
        config = load_config()
        self.history_dir = history_dir or config["analysis"]["history_dir"]

    def _company_dir(self, company: str) -> str:
        """Normalize company name to safe directory path."""
        safe_name = re.sub(r"[^\w]", "_", company.lower()).strip("_")
        path = os.path.join(self.history_dir, safe_name)
        os.makedirs(path, exist_ok=True)
        return path

    def save(self, company: str, data: Dict[str, Any]) -> str:
        """Persist a snapshot for today. Returns the filepath written."""
        company_dir = self._company_dir(company)
        date_str = datetime.now().strftime("%Y-%m-%d")
        filepath = os.path.join(company_dir, f"{date_str}.json")

        payload = {
            "company": company,
            "date": date_str,
            "timestamp": datetime.now().isoformat(),
            **data,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info(f"Snapshot saved: {filepath}")
        return filepath

    def load(self, company: str) -> Optional[Dict[str, Any]]:
        """
        Load the most recent historical snapshot for a company.
        Returns None if no history exists (first run).
        """
        company_dir = self._company_dir(company)
        snapshots = sorted(
            [f for f in os.listdir(company_dir) if f.endswith(".json")],
            reverse=True,
        )

        # Skip today's snapshot — we want the previous one for delta
        today = datetime.now().strftime("%Y-%m-%d")
        previous = [s for s in snapshots if not s.startswith(today)]

        if not previous:
            logger.info(f"No historical baseline found for {company}.")
            return None

        latest_file = os.path.join(company_dir, previous[0])
        with open(latest_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def compute_delta(
        self,
        historical: Optional[Dict[str, Any]],
        current: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Thin wrapper that delegates to TrendCalculator but lives here
        so callers only need one import.
        """
        from utils.trend_calculator import TrendCalculator
        return TrendCalculator.compute_delta(current, historical)

    def list_companies(self) -> List[str]:
        """List all companies that have stored history."""
        if not os.path.exists(self.history_dir):
            return []
        return [d for d in os.listdir(self.history_dir)
                if os.path.isdir(os.path.join(self.history_dir, d))]
