import logging
from langchain_tavily import TavilySearch

logger = logging.getLogger(__name__)


class NewsFetcher:
    """
    Fetches news and funding intelligence for a given company.

    Uses two separate TavilySearch clients:
    - topic="news"    → recent product/research announcements
    - topic="general" → funding rounds, acquisitions (not always indexed as "news")
    """

    def __init__(self):
        self.search = TavilySearch(
            topic="news",
            include_answer="advanced",
            max_results=5,
        )
        self.general_search = TavilySearch(
            topic="general",
            include_answer="advanced",
            max_results=5,
        )

    def get_ai_news(self, company: str) -> str:
        """Fetch recent AI product launches, research publications, and announcements."""
        query = f"{company} AI product launch research announcement 2025"
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
            logger.warning(f"News fetch failed for {company}: {e}")
            return f"No recent news available for {company}."

    def get_funding_and_partnerships(self, company: str) -> str:
        """Fetch recent funding rounds, acquisitions, and strategic partnerships."""
        query = f"{company} funding investment acquisition partnership 2025"
        try:
            result = self.general_search.invoke({"query": query})
            if isinstance(result, dict) and result.get("answer"):
                return result["answer"]
            if isinstance(result, list):
                return "\n\n".join(
                    r.get("content", "") for r in result if isinstance(r, dict)
                )
            return str(result)
        except Exception as e:
            logger.warning(f"Funding search failed for {company}: {e}")
            return f"No funding data available for {company}."
