from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TrendCalculator:
    """
    Computes posting velocity deltas between current and historical snapshots.

    All math is isolated here so it can be unit tested independently of
    the LangGraph pipeline.
    """

    @staticmethod
    def compute_delta(
        current: Dict[str, Any],
        historical: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute delta between current and historical job data snapshots.

        Returns a dict with:
        - previous_count: int
        - current_count: int  (estimated from text length heuristic if no count)
        - absolute_change: int
        - percentage_change: float
        - direction: "up" | "down" | "flat" | "new"
        """
        if not historical:
            return {
                "previous_count": 0,
                "current_count": 0,
                "absolute_change": 0,
                "percentage_change": 0.0,
                "direction": "new",
                "note": "No historical baseline — this is the first snapshot.",
            }

        prev = historical.get("estimated_posting_count", 0)
        curr = current.get("estimated_posting_count", 0)

        absolute_change = curr - prev
        if prev == 0:
            percentage_change = 100.0 if curr > 0 else 0.0
        else:
            percentage_change = round((absolute_change / prev) * 100, 1)

        if abs(percentage_change) < 5:
            direction = "flat"
        elif percentage_change > 0:
            direction = "up"
        else:
            direction = "down"

        return {
            "previous_count": prev,
            "current_count": curr,
            "absolute_change": absolute_change,
            "percentage_change": percentage_change,
            "direction": direction,
        }

    @staticmethod
    def estimate_posting_count(postings_text: str) -> int:
        """
        Heuristic count estimate from raw posting text.
        Counts occurrences of common job posting markers.
        """
        if not postings_text:
            return 0
        markers = ["engineer", "scientist", "researcher", "manager", "analyst"]
        text_lower = postings_text.lower()
        return sum(text_lower.count(m) for m in markers)
