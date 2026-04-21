import logging
import requests
from langchain_tavily import TavilySearch

logger = logging.getLogger(__name__)

# Tier 1: known Greenhouse board slugs (no API key; free structured data)
GREENHOUSE_SLUGS = {
    "Anthropic": "anthropic",
    "OpenAI": "openai",
    "Mistral": "mistralai",
    "Cohere": "cohere",
    "Hugging Face": "huggingface",
}

# Keywords to keep only AI/ML-relevant jobs when using Greenhouse
_AI_TITLE_KEYWORDS = [
    "machine learning",
    "research",
    "llm",
    "ml ",
    "ai ",
    "data scientist",
    "inference",
    "alignment",
]


class JobFetcher:
    """
    Fetches job posting data for a given company using a three-tier strategy:
    Greenhouse API (when slug known) → Tavily (targeted) → broad fallback.
    """

    def __init__(self):
        self.search = TavilySearch(
            topic="general",
            include_answer="advanced",
            max_results=5,
        )

    def search_ai_postings(self, company: str) -> str:
        """
        Search for active AI/ML job postings at a company.
        Tier 1: Greenhouse API if we have a slug. Tier 2: Tavily targeted query. Tier 3: fallback.
        """
        # --- Tier 1: Greenhouse API (structured, free, no key) ---
        # Use known board slugs so we get real job listings without search noise.
        slug = GREENHOUSE_SLUGS.get(company)
        if slug:
            try:
                url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true"
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                jobs = data.get("jobs", [])
                # Filter to jobs whose title contains any AI/ML keyword
                matching = []
                for job in jobs:
                    title = (job.get("title") or "").lower()
                    if any(kw in title for kw in _AI_TITLE_KEYWORDS):
                        matching.append(job)
                # First 10: title + first 600 chars of description
                parts = []
                for job in matching[:10]:
                    title = job.get("title", "")
                    content = (job.get("content") or "")[:600]
                    parts.append(f"{title}\n{content}")
                if parts:
                    return "\n\n".join(parts)
            except Exception as e:
                logger.warning(f"Greenhouse Tier 1 failed for {company}: {e}")

        # --- Tier 2: Tavily with a precise targeted query ---
        # Targets job board pages (greenhouse, lever, ashby) and AI role terms.
        query = (
            f'"{company}" (site:greenhouse.io OR site:lever.co OR site:ashby.com OR site:jobs.lever.co) '
            f'("machine learning" OR "research scientist" OR "LLM" OR "AI engineer") 2025'
        )
        try:
            result = self.search.invoke({"query": query})
            if isinstance(result, dict) and result.get("answer"):
                return result["answer"]
            if isinstance(result, list):
                return "\n\n".join(
                    r.get("content", "") for r in result if isinstance(r, dict)
                )
            return str(result)
        except Exception as e:
            logger.warning(f"Tavily Tier 2 failed for {company}: {e}")

        # --- Tier 3: Broad fallback ---
        # When Tier 2 raises, use a generic search so we still return something.
        return self._fallback_search(company)

    def get_posting_velocity(self, company: str, timeframe_days: int = 30) -> str:
        """
        Estimate recent posting velocity — how many AI roles has this company
        posted in the last N days?
        """
        query = (
            f'"{company}" hiring AI machine learning jobs posted last {timeframe_days} days 2025'
        )
        try:
            result = self.search.invoke({"query": query})
            if isinstance(result, dict) and result.get("answer"):
                return result["answer"]
            return str(result)
        except Exception as e:
            logger.warning(f"Velocity search failed for {company}: {e}")
            return f"Could not retrieve velocity data for {company}."

    def _fallback_search(self, company: str) -> str:
        """Broader fallback when site-specific search fails."""
        query = f"{company} AI machine learning job openings hiring 2025"
        try:
            result = self.search.invoke({"query": query})
            return result.get("answer", str(result)) if isinstance(result, dict) else str(result)
        except Exception as e:
            logger.error(f"Fallback job search failed for {company}: {e}")
            return f"No job data available for {company}."
