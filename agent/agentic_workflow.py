import json
import logging
import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send

from prompt_library.prompts import NARRATIVE_PROMPT, ORCHESTRATOR_PROMPT
from tools.company_news_tool import CompanyNewsTool
from tools.job_search_tool import JobSearchTool
from tools.skill_extractor_tool import SkillExtractorTool
from tools.trend_delta_tool import TrendDeltaTool
from utils.arxiv_fetcher import fetch_recent_papers
from utils.data_store import DataStore
from utils.model_loader import ModelLoader
from utils.skill_taxonomy import compute_intent_score
from utils.trend_calculator import TrendCalculator

logger = logging.getLogger(__name__)


# ── Reducers ──────────────────────────────────────────────────────────────────

def merge_dicts(a: Dict, b: Dict) -> Dict:
    """State reducer: merges two dicts together.

    Used on fields that receive parallel writes from concurrent
    company_researcher node executions via Send().
    Without this, parallel workers would overwrite each other's results.
    """
    return {**a, **b}


# ── State ─────────────────────────────────────────────────────────────────────

class CompanyResearchInput(TypedDict):
    """The slice of state sent to each parallel company_researcher node via Send()."""
    company: str
    timeframe_days: int


class TalentSignalState(TypedDict):
    """Full pipeline state. Each field is updated by a specific node.

    Fields annotated with merge_dicts support concurrent writes from
    parallel company_researcher executions.
    """
    # Input
    companies: List[str]
    timeframe_days: int

    # Parallel research output
    company_job_data:  Annotated[Dict[str, Any], merge_dicts]
    company_news_data: Annotated[Dict[str, Any], merge_dicts]

    # Sequential downstream
    extracted_skills:   Dict[str, Any]
    trend_deltas:       Dict[str, Any]
    company_narratives: Dict[str, str]
    final_report:       str

    # Metadata
    messages: Annotated[List[BaseMessage], add_messages]
    error:    Optional[str]


# ── GraphBuilder ──────────────────────────────────────────────────────────────

class GraphBuilder:
    """Builds and compiles the HireSignal LangGraph pipeline.

    Usage:
        builder = GraphBuilder(model_provider="groq")
        graph = builder()            # compiles and returns the graph
        result = graph.invoke({...}) # run synchronously
        for event in graph.stream({...}):  # or stream events
            ...

    BUG FIX: In the original travel planner this was rebuilt on every API request.
    Compile once at startup (in FastAPI lifespan) and reuse.
    """

    def __init__(self, model_provider: str = "groq"):
        self.model_loader = ModelLoader(model_provider=model_provider)
        self.llm          = self.model_loader.load_llm()
        self.data_store   = DataStore()

        # Tool classes (Module 5 wrappers around Module 4 utils)
        self.job_search_tool = JobSearchTool()
        self.news_tool       = CompanyNewsTool()
        self.skill_tool      = SkillExtractorTool()
        self.trend_tool      = TrendDeltaTool()

        self.graph = None

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def validate_input(self, state: TalentSignalState) -> Dict:
        """Normalizes company names, sets defaults, initializes all state fields."""
        companies = state.get("companies", [])
        if not companies:
            logger.error("No companies provided.")
            return {"error": "No companies provided."}

        companies = [c.strip() for c in companies if c.strip()]
        timeframe = state.get("timeframe_days", 30)
        logger.info(f"Pipeline started: {len(companies)} companies — {companies}")

        return {
            "companies":         companies,
            "timeframe_days":    timeframe,
            "company_job_data":  {},
            "company_news_data": {},
            "extracted_skills":  {},
            "trend_deltas":      {},
            "company_narratives":{},
            "final_report":      "",
            "error":             None,
            "messages":          [HumanMessage(content=f"Analyzing: {companies}")],
        }

    def coordinator(self, state: TalentSignalState) -> Dict:
        """Passthrough node. Exists as a clean routing checkpoint between
        the validation error gate and the Send() fan-out.
        """
        return {}

    def company_researcher(self, state: CompanyResearchInput) -> Dict:
        """Parallel research node — one execution per company via Send().

        Receives a minimal CompanyResearchInput state (not the full pipeline state).
        Writes back to company_job_data and company_news_data via merge_dicts reducer.
        """
        company   = state["company"]
        timeframe = state.get("timeframe_days", 30)
        logger.info(f"Researching: {company}")

        # Job data
        job_results = {}
        try:
            postings_tool = next(
                t for t in self.job_search_tool.tool_list
                if t.name == "search_ai_job_postings"
            )
            velocity_tool = next(
                t for t in self.job_search_tool.tool_list
                if t.name == "get_posting_velocity"
            )
            job_results["postings"] = postings_tool.invoke({"company": company})
            job_results["velocity"] = velocity_tool.invoke({
                "company": company, "timeframe_days": timeframe
            })
            job_results["estimated_posting_count"] = TrendCalculator.estimate_posting_count(
                job_results["postings"]
            )
        except Exception as e:
            logger.warning(f"Job fetch failed for {company}: {e}")
            job_results = {
                "error": str(e),
                "postings": "",
                "velocity": "",
                "estimated_posting_count": 0,
            }

        # News data
        news_results = {}
        try:
            news_tool    = next(t for t in self.news_tool.tool_list if t.name == "get_company_ai_news")
            funding_tool = next(t for t in self.news_tool.tool_list if t.name == "get_funding_and_partnerships")
            news_results["news"]    = news_tool.invoke({"company": company})
            news_results["funding"] = funding_tool.invoke({"company": company})
        except Exception as e:
            logger.warning(f"News fetch failed for {company}: {e}")
            news_results = {"error": str(e), "news": "", "funding": ""}

        # arXiv papers: one line per paper as "TITLE (DATE): SUMMARY"
        try:
            papers = fetch_recent_papers(company)
            news_results["papers"] = "\n".join(
                f"{p['title']} ({p['published']}): {p['summary']}" for p in papers
            )
        except Exception as e:
            logger.warning(f"arXiv fetch failed for {company}: {e}")
            news_results["papers"] = ""

        return {
            "company_job_data":  {company: job_results},
            "company_news_data": {company: news_results},
        }

    def extract_skills(self, state: TalentSignalState) -> Dict:
        """Runs skill extraction on all collected job posting data."""
        job_data  = state.get("company_job_data", {})
        extracted = {}

        extract_tool = next(
            t for t in self.skill_tool.tool_list
            if t.name == "extract_ai_skills"
        )

        for company, data in job_data.items():
            postings_text = data.get("postings", "")
            try:
                raw = extract_tool.invoke({"postings_text": postings_text})
                extracted[company] = json.loads(raw)
            except Exception as e:
                logger.warning(f"Skill extraction failed for {company}: {e}")
                extracted[company] = {
                    "top_skills": [], "domains": [], "role_types": [], "team_signals": []
                }

        logger.info(f"Skills extracted for {len(extracted)} companies.")
        return {"extracted_skills": extracted}

    def analyze_trends(self, state: TalentSignalState) -> Dict:
        """Computes deltas vs historical snapshots, scores intent, saves snapshots."""
        job_data    = state.get("company_job_data", {})
        skills_data = state.get("extracted_skills", {})
        news_data   = state.get("company_news_data", {})
        deltas = {}

        for company in state["companies"]:
            current_jobs   = job_data.get(company, {})
            current_skills = skills_data.get(company, {})
            current_news   = news_data.get(company, {})

            historical = self.data_store.load(company)
            delta      = self.data_store.compute_delta(historical, current_jobs)

            # Funding keyword detection
            funding_text = current_news.get("funding", "").lower()
            has_funding  = any(
                word in funding_text
                for word in ["raised", "funding", "series", "investment", "acquired", "partnership"]
            )

            papers_text = current_news.get("papers", "")
            has_recent_papers = len(papers_text) > 50

            intent_score = compute_intent_score(
                postings_text=current_jobs.get("postings", ""),
                domains=current_skills.get("domains", []),
                has_funding_news=has_funding,
                has_recent_papers=has_recent_papers,
            )

            deltas[company] = {
                "posting_delta":    delta,
                "intent_score":     intent_score,
                "top_skills":       current_skills.get("top_skills", []),
                "ai_domains":       current_skills.get("domains", []),
                "role_types":       current_skills.get("role_types", []),
                "team_signals":     current_skills.get("team_signals", []),
                "has_funding_news": has_funding,
                "recent_papers":    papers_text,
            }

            self.data_store.save(company, current_jobs)

        logger.info("Trend analysis complete.")
        return {"trend_deltas": deltas}

    def generate_narratives(self, state: TalentSignalState) -> Dict:
        """LLM generates per-company intelligence narratives."""
        narratives = {}

        for company in state["companies"]:
            job_data   = state["company_job_data"].get(company, {})
            news_data  = state["company_news_data"].get(company, {})
            delta_data = state["trend_deltas"].get(company, {})

            prompt = NARRATIVE_PROMPT.format(
                company       = company,
                intent_score  = delta_data.get("intent_score", "N/A"),
                job_count     = len(delta_data.get("role_types", [])),
                job_postings  = job_data.get("postings", "No data.")[:2000],
                velocity      = job_data.get("velocity", "No data."),
                recent_papers = delta_data.get("recent_papers", "No papers found."),
                news          = news_data.get("news", "No data.")[:1500],
                funding       = news_data.get("funding", "No data.")[:1000],
                top_skills    = ", ".join(delta_data.get("top_skills", [])) or "None detected",
                ai_domains    = ", ".join(delta_data.get("ai_domains", [])) or "None detected",
                role_types    = ", ".join(delta_data.get("role_types", [])) or "None detected",
                team_signals  = ", ".join(delta_data.get("team_signals", [])) or "None detected",
            )

            try:
                response = self.llm.invoke([SystemMessage(content=prompt)])
                narratives[company] = response.content
            except Exception as e:
                narratives[company] = f"*Narrative generation failed: {e}*"

        return {"company_narratives": narratives}

    def synthesize_report(self, state: TalentSignalState) -> Dict:
        """Final synthesis: LLM integrates per-company narratives into a ranked brief."""
        narratives = state.get("company_narratives", {})
        deltas     = state.get("trend_deltas", {})

        ranked_companies = sorted(
            state["companies"],
            key=lambda c: deltas.get(c, {}).get("intent_score", 0),
            reverse=True,
        )

        leaderboard = "\n".join(
            f"| {c} | {deltas.get(c, {}).get('intent_score', 'N/A')}/100 | "
            f"{', '.join(deltas.get(c, {}).get('ai_domains', [])[:2]) or 'N/A'} |"
            for c in ranked_companies
        )

        narratives_block = "\n\n---\n\n".join(
            f"## {c}\n{narratives.get(c, '*No narrative available.*')}"
            for c in ranked_companies
        )

        synthesis_prompt = f"""{ORCHESTRATOR_PROMPT}

## Intent Score Leaderboard
| Company | Score | Top Domains |
|---------|-------|-------------|
{leaderboard}

---

## Individual Company Analyses

{narratives_block}

---

Using the data above, write the final HireSignal Intelligence Brief (four sections as specified in the rules).
"""

        try:
            response    = self.llm.invoke([SystemMessage(content=synthesis_prompt)])
            final_report = response.content
        except Exception as e:
            final_report = f"*Synthesis failed: {e}*\n\n" + narratives_block

        return {
            "final_report": final_report,
            "messages":     [SystemMessage(content=final_report)],
        }

    # ── Routing ───────────────────────────────────────────────────────────────

    def route_after_validation(self, state: TalentSignalState) -> str:
        """Error gate: routes to END if validation failed, coordinator otherwise."""
        if state.get("error"):
            logger.error(f"Routing to error end: {state['error']}")
            return "error_end"
        return "continue"

    def fan_out_companies(self, state: TalentSignalState) -> List[Send]:
        """Fan-out routing: one Send per company → parallel company_researcher."""
        return [
            Send(
                "company_researcher",
                {
                    "company":        company,
                    "timeframe_days": state["timeframe_days"],
                },
            )
            for company in state["companies"]
        ]

    # ── Graph ─────────────────────────────────────────────────────────────────

    def build_graph(self):
        """Assemble, wire, and compile the StateGraph.

        Call once at startup. Reuse the compiled graph for all requests.
        """
        g = StateGraph(TalentSignalState)

        # Register nodes
        g.add_node("validate_input",      self.validate_input)
        g.add_node("coordinator",         self.coordinator)
        g.add_node("company_researcher",  self.company_researcher)
        g.add_node("extract_skills",      self.extract_skills)
        g.add_node("analyze_trends",      self.analyze_trends)
        g.add_node("generate_narratives", self.generate_narratives)
        g.add_node("synthesize_report",   self.synthesize_report)

        # Entry
        g.add_edge(START, "validate_input")

        # Error gate
        g.add_conditional_edges(
            "validate_input",
            self.route_after_validation,
            {"error_end": END, "continue": "coordinator"},
        )

        # Fan-out
        g.add_conditional_edges(
            "coordinator",
            self.fan_out_companies,
            ["company_researcher"],
        )

        # Convergence + sequential pipeline
        g.add_edge("company_researcher",  "extract_skills")
        g.add_edge("extract_skills",      "analyze_trends")
        g.add_edge("analyze_trends",      "generate_narratives")
        g.add_edge("generate_narratives", "synthesize_report")
        g.add_edge("synthesize_report",   END)

        self.graph = g.compile()
        logger.info("LangGraph pipeline compiled successfully.")
        return self.graph

    def __call__(self):
        return self.build_graph()
