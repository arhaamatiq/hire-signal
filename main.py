import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent.agentic_workflow import GraphBuilder
from utils.report_exporter import export_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Shared state ─────────────────────────────────────────────────────────────
_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Compile the LangGraph pipeline ONCE at startup.

    BUG FIX: In the original, GraphBuilder was instantiated inside the route
    handler — which means the entire graph (LLM clients, tool classes, fetchers)
    was rebuilt on every request. Under load this is expensive and rate-limits
    API key initialization.

    Compiling once at startup and reusing the compiled graph fixes this.
    """
    global _graph
    logger.info("Starting up — compiling LangGraph pipeline...")
    builder = GraphBuilder(model_provider="groq")
    _graph  = builder()
    logger.info("Pipeline ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="HireSignal",
    description="Agentic AI hiring intelligence pipeline — decodes who is building what.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response models ──────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    companies: List[str] = Field(
        default=["Anthropic", "OpenAI", "Google DeepMind"],
        description="List of AI companies to analyze.",
        min_length=1,
    )
    timeframe_days: int = Field(
        default=30,
        ge=7,
        le=90,
        description="How many days back to look for posting velocity.",
    )
    model_provider: Optional[str] = Field(
        default="groq",
        description="LLM provider: 'groq' or 'openai'.",
    )


class AnalyzeResponse(BaseModel):
    final_report: str
    companies_analyzed: List[str]
    intent_scores: dict
    company_narratives: Optional[dict] = None
    company_job_data: Optional[dict] = None
    company_news_data: Optional[dict] = None
    report_filepath: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "pipeline": "compiled" if _graph else "not ready"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Run the full HireSignal pipeline for the requested companies.
    Returns a ranked intelligence report.
    """
    if _graph is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    logger.info(f"Received request: {request.companies}")

    try:
        result = _graph.invoke({
            "companies":      request.companies,
            "timeframe_days": request.timeframe_days,
        })
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])

    # Extract intent scores for the response
    trend_deltas  = result.get("trend_deltas", {})
    intent_scores = {
        company: data.get("intent_score", 0)
        for company, data in trend_deltas.items()
    }

    # BUG FIX: export_report was defined but never called — reports were discarded.
    # Now called after every successful run.
    report_filepath = None
    try:
        report_filepath = export_report(
            report_text=result.get("final_report", ""),
            companies=request.companies,
        )
    except Exception as e:
        logger.warning(f"Report export failed (non-fatal): {e}")

    return AnalyzeResponse(
        final_report        = result.get("final_report", ""),
        companies_analyzed  = request.companies,
        intent_scores       = intent_scores,
        company_narratives  = result.get("company_narratives"),
        company_job_data    = result.get("company_job_data"),
        company_news_data   = result.get("company_news_data"),
        report_filepath     = report_filepath,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
