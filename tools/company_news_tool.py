from typing import List
from langchain_core.tools import tool
from utils.news_fetcher import NewsFetcher


class CompanyNewsTool:
    """LangChain tool wrappers for company news and intelligence."""

    def __init__(self):
        self.fetcher = NewsFetcher()
        self.tool_list = self._setup_tools()

    def _setup_tools(self) -> List:

        @tool
        def get_company_ai_news(company: str) -> str:
            """Fetch recent AI product launches, research publications, and
            strategic announcements for a company.
            Returns a summary of the last 30 days.
            """
            return self.fetcher.get_ai_news(company)

        @tool
        def get_funding_and_partnerships(company: str) -> str:
            """Fetch recent funding rounds, acquisitions, and strategic
            partnerships. These are leading indicators of upcoming AI
            product investment and scale.
            """
            return self.fetcher.get_funding_and_partnerships(company)

        return [get_company_ai_news, get_funding_and_partnerships]
