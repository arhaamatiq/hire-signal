import json
import logging
from typing import List
from langchain_core.tools import tool
from utils.trend_calculator import TrendCalculator
from utils.skill_taxonomy import compute_intent_score
from utils.data_store import DataStore

logger = logging.getLogger(__name__)


class TrendDeltaTool:
    """Tool wrappers for trend calculation and intent scoring."""

    def __init__(self):
        self.calculator = TrendCalculator()
        self.data_store = DataStore()
        self.tool_list = self._setup_tools()

    def _setup_tools(self) -> List:

        @tool
        def calculate_hiring_delta(company: str, current_posting_text: str) -> str:
            """Compare current AI job postings to the most recent historical
            snapshot for this company. Returns a JSON delta report with
            direction ("up"/"down"/"flat"/"new") and percentage change.
            """
            estimated_count = TrendCalculator.estimate_posting_count(current_posting_text)
            current_data = {"estimated_posting_count": estimated_count}

            historical = self.data_store.load(company)
            delta = TrendCalculator.compute_delta(current_data, historical)

            return json.dumps(delta)

        @tool
        def score_ai_intent(
            company: str,
            posting_delta_json: str,
            domains_json: str,
            has_funding_news: bool = False,
            postings_text: str = "",
            has_recent_papers: bool = False,
        ) -> str:
            """Compute a 0-100 AI intent score for a company based on:
            - Job posting text (volume of AI role keywords)
            - AI domain coverage (what are they hiring for?)
            - Recent funding/partnership news and arXiv papers.
            Optional: postings_text and has_recent_papers for full scoring when available.
            Returns a JSON object with score and explanation.
            """
            try:
                domains = json.loads(domains_json) if isinstance(domains_json, str) else domains_json
            except Exception:
                domains = []

            score, breakdown = compute_intent_score(
                postings_text=postings_text,
                domains=domains,
                has_funding_news=has_funding_news,
                has_recent_papers=has_recent_papers,
            )

            if score >= 75:
                label = "🔴 HIGH — Likely scaling an AI product imminently"
            elif score >= 55:
                label = "🟡 MEDIUM — Active AI investment, worth monitoring"
            elif score >= 35:
                label = "🟢 LOW-MEDIUM — Steady AI hiring, no surge detected"
            else:
                label = "⚪ LOW — Minimal AI hiring signal"

            return json.dumps({
                "company":      company,
                "intent_score": score,
                "label":        label,
                "breakdown":    breakdown,
                "components": {
                    "domain_count":      len(domains),
                    "funding_bonus":     has_funding_news,
                    "has_recent_papers": has_recent_papers,
                },
            })

        return [calculate_hiring_delta, score_ai_intent]
