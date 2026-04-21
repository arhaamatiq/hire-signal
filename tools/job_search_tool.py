from typing import List
from langchain_core.tools import tool
from utils.job_fetcher import JobFetcher


class JobSearchTool:
    """LangChain tool wrappers for job posting research.

    Follows the tool/utils separation pattern:
    - tools/ contains thin @tool-decorated wrappers
    - utils/ contains the actual logic and API calls
    """

    def __init__(self):
        self.fetcher = JobFetcher()
        self.tool_list = self._setup_tools()

    def _setup_tools(self) -> List:

        @tool
        def search_ai_job_postings(company: str) -> str:
            """Search for active AI and machine learning job postings at a company.
            Returns job titles, descriptions, and required skills from LinkedIn,
            Greenhouse, and Lever.
            """
            return self.fetcher.search_ai_postings(company)

        @tool
        def get_posting_velocity(company: str, timeframe_days: int = 30) -> str:
            """Estimate how many AI/ML roles a company has posted in the last N days.
            Useful for detecting hiring surges that precede product launches.
            """
            return self.fetcher.get_posting_velocity(company, timeframe_days)

        return [search_ai_job_postings, get_posting_velocity]
