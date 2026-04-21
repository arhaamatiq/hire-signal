"""
HireSignal — Streamlit frontend for the AI hiring intelligence pipeline.
Calls the FastAPI backend and displays results in a clean, tabbed dashboard.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests
import streamlit as st

DEFAULT_COMPANIES = ["Anthropic", "OpenAI", "Google DeepMind", "Meta AI", "Mistral"]
DEFAULT_TIMEFRAME_DAYS = 30
DEFAULT_API_BASE = "http://localhost:8000"


def _sanitize_report(text: str) -> str:
    """Remove common API/metadata lines that sometimes appear in LLM output (e.g. Groq billing links)."""
    if not text or not text.strip():
        return text
    lines = text.split("\n")
    out = []
    for line in lines:
        s = line.strip()
        if not s:
            out.append(line)
            continue
        if "console.groq.com" in s or "groq.com/settings" in s:
            continue
        if s.startswith("org_") and len(s) > 20:
            continue
        if s in ("on_demand", "versatile") or (s.startswith("llama-3") and len(s) < 30):
            continue
        out.append(line)
    return "\n".join(out)


def normalize_api_url(url: str) -> str:
    """Use http for localhost (uvicorn does not serve HTTPS). Avoids unreachable backend."""
    u = (url or "").strip().rstrip("/")
    if u.startswith("https://localhost") or u.startswith("https://127.0.0.1"):
        u = "http" + u[5:]
    return u or DEFAULT_API_BASE


def load_default_companies():
    try:
        from utils.config_loader import load_config
        config = load_config()
        return config.get("companies", {}).get("defaults", DEFAULT_COMPANIES)
    except Exception:
        return DEFAULT_COMPANIES


# Comprehensive styling: light background + dark text everywhere so output is always readable
def apply_custom_css():
    st.markdown("""
    <style>
    /* App shell: light background */
    .stApp, [data-testid="stAppViewContainer"] { background: #ffffff !important; }
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 960px; }

    /* All markdown and text: dark, readable */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
    [data-testid="stMarkdown"] p, [data-testid="stMarkdown"] li, [data-testid="stMarkdown"] span,
    .main .stMarkdown, .main p, .main li, .main span { color: #1e293b !important; }
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 { color: #0f172a !important; }

    /* Metrics and labels */
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
    .stMetric label, .stMetric [data-testid="stMetricValue"] { color: #1e293b !important; }
    .stCaption { color: #475569 !important; }

    /* Tabs and expanders: readable text */
    .stTabs [data-baseweb="tab-list"] button { color: #1e293b !important; }
    .streamlit-expanderContent p, .streamlit-expanderContent li, .streamlit-expanderContent span { color: #1e293b !important; }

    /* Info and alert boxes: dark text on tinted background */
    .stAlert, [data-testid="stAlert"] { color: #1e293b !important; }
    .stAlert a { color: #4f46e5 !important; }

    /* Code blocks: dark text on light grey (readable) */
    .stCodeBlock code, code { color: #1e293b !important; background: #f1f5f9 !important; }

    /* Sidebar: light background, dark text */
    [data-testid="stSidebar"] { background: #f8fafc !important; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: #1e293b !important; }

    /* Hero */
    .hero { text-align: center; margin-bottom: 1.5rem; }
    .hero h1 { font-size: 1.75rem; color: #1e293b !important; margin-bottom: 0.25rem; }
    .hero p { color: #475569 !important; font-size: 0.95rem; }

    /* Progress bar */
    .stProgress > div > div > div { background: #e2e8f0 !important; }
    .stProgress > div > div > div > div { background: #4f46e5 !important; }

    /* Report and markdown content: ensure body text and links are readable */
    .main .stMarkdown p, .main .stMarkdown li, .main .stMarkdown div { color: #1e293b !important; }
    .main a { color: #4f46e5 !important; }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


def intent_label(score: int) -> str:
    """Return a short label for the intent score band (used under the progress bar)."""
    if score >= 75:
        return "High intent"
    if score >= 55:
        return "Medium intent"
    if score >= 35:
        return "Low–medium"
    return "Low intent"


def check_health(base_url: str) -> tuple[bool, str]:
    """Try base_url, then 127.0.0.1 if base uses localhost. Returns (ok, url_used)."""
    url = (base_url or "").strip().rstrip("/")
    if not url:
        return False, ""
    urls_to_try = [url]
    if "localhost" in url:
        alt = url.replace("localhost", "127.0.0.1")
        if alt != url:
            urls_to_try.append(alt)
    for u in urls_to_try:
        try:
            r = requests.get(f"{u}/health", timeout=10)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return True, u
        except Exception:
            continue
    return False, url


def run_analyze(base_url: str, companies: list[str], timeframe_days: int, model_provider: str | None):
    base_url = (base_url or "").strip().rstrip("/")
    url = f"{base_url}/analyze"
    payload = {"companies": companies, "timeframe_days": timeframe_days}
    if model_provider:
        payload["model_provider"] = model_provider
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


def main():
    st.set_page_config(
        page_title="HireSignal — AI Hiring Intelligence",
        page_icon="🧠",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    apply_custom_css()

    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        api_base_raw = st.text_input(
            "API base URL",
            value=os.environ.get("HIRESIGNAL_API_URL", DEFAULT_API_BASE),
            help="Use http://localhost:8000 (not https) for local backend.",
        )
        api_base = normalize_api_url(api_base_raw)
        model_provider = st.selectbox(
            "Model provider",
            options=["groq", "openai"],
            index=0,
            help="LLM used for synthesis.",
        )
        st.markdown("---")
        healthy, url_used = check_health(api_base)
        if healthy:
            st.success(f"Backend connected at {url_used}")
        else:
            st.error(
                "Backend unreachable. In a terminal, from the **project root** run: "
                "`./scripts/start_backend.sh` or `uvicorn main:app --reload --host 0.0.0.0 --port 8000`"
            )
            if api_base_raw.strip().lower().startswith("https://"):
                st.warning("Use **http** (not https) for local — the API does not use SSL.")
        # Use the URL that passed health check, or normalized input for the actual request
        api_base_effective = url_used if healthy else api_base

    st.markdown(
        '<div class="hero"><h1>🧠 HireSignal</h1>'
        '<p>Decode AI lab hiring patterns into actionable intelligence — who\'s scaling, what they\'re building, what to watch.</p></div>',
        unsafe_allow_html=True,
    )

    default_companies = load_default_companies()
    companies = st.multiselect(
        "Companies to analyze",
        options=default_companies + ["Anthropic", "OpenAI", "Google DeepMind", "Meta AI", "Mistral", "Cohere", "Scale AI", "Hugging Face", "Stability AI"],
        default=default_companies[:3],
        help="Select one or more AI/ML companies.",
    )
    timeframe_days = st.slider(
        "Lookback window (days)",
        min_value=7,
        max_value=90,
        value=DEFAULT_TIMEFRAME_DAYS,
        help="How many days back to consider for posting velocity.",
    )
    run_btn = st.button("Run analysis", type="primary", use_container_width=True)

    if run_btn:
        if not companies:
            st.warning("Select at least one company.")
            st.stop()
        if not healthy:
            st.error("Start the backend first, then try again.")
            st.stop()

        with st.spinner("Running pipeline (job search → skills → trends → narratives → report)…"):
            try:
                result = run_analyze(api_base_effective, companies, timeframe_days, model_provider)
            except requests.exceptions.ConnectionError as e:
                st.error(
                    "Could not reach the backend. From project root run: `./scripts/start_backend.sh`"
                )
                st.stop()
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
                if hasattr(e, "response") and e.response is not None:
                    try:
                        st.code(e.response.text)
                    except Exception:
                        pass
                st.stop()

        # --- Build ranked list and score helper (same order used everywhere below) ---
        intent_scores = result.get("intent_scores", {})
        companies_analyzed = result.get("companies_analyzed", list(intent_scores.keys()))

        def _score(c):
            s = intent_scores.get(c, 0)
            return int(s) if s is not None and isinstance(s, (int, float)) else 0

        ranked = sorted(companies_analyzed, key=_score, reverse=True)

        # --- Pipeline summary: what was actually fetched (builds user confidence) ---
        job_data = result.get("company_job_data") or {}
        news_data = result.get("company_news_data") or {}
        paper_count = sum(1 for c in ranked if len((news_data.get(c) or {}).get("papers", "") or "") > 50)
        posting_count = sum(((job_data.get(c) or {}).get("postings", "") or "").count("\n") for c in ranked)
        st.info(
            f"Pipeline fetched data for **{len(ranked)}** companies · **{paper_count}** had arXiv papers · "
            f"**~{posting_count}** job posting signals found"
        )

        # --- Intent score leaderboard: one clear row per company (metric + bar + label) ---
        st.markdown("#### Intent score leaderboard")
        for company in ranked:
            score = _score(company)
            with st.container():
                col_name, col_bar = st.columns([2, 3])
                with col_name:
                    st.metric(label=company, value=f"{score}/100")
                with col_bar:
                    st.progress(score / 100.0)
                    st.caption(intent_label(score))

        if result.get("report_filepath"):
            st.caption(f"Report saved: {result['report_filepath']}")

        st.markdown("---")

        # --- Tabbed layout: Intelligence Brief | Per-Company Briefs | Raw Data ---
        tab_brief, tab_per_company, tab_raw = st.tabs(["Intelligence Brief", "Per-Company Briefs", "Raw Data"])

        with tab_brief:
            # Render the report in a single container; strip any API metadata that leaked into the LLM output
            report_text = _sanitize_report(result.get("final_report", ""))
            if report_text.strip():
                st.markdown("---")
                # Wrap in a container so layout doesn't fragment; markdown renders headers/bullets/bold
                with st.container():
                    st.markdown(report_text)
            else:
                st.markdown("_No report generated._")

        with tab_per_company:
            narratives = result.get("company_narratives") or {}
            for i, company in enumerate(ranked):
                score = _score(company)
                # Top company expanded by default; rest collapsed
                with st.expander(f"{company} — {score}/100", expanded=(i == 0)):
                    st.markdown(narratives.get(company, "_No narrative._"))

        with tab_raw:
            # Show exactly what the pipeline fetched so users can judge data quality
            for company in ranked:
                with st.expander(company):
                    jd = (job_data.get(company) or {}).get("postings", "") or ""
                    nd = news_data.get(company) or {}
                    raw_news = nd.get("news", "") or ""
                    raw_papers = nd.get("papers", "") or ""
                    st.markdown("**💼 Job postings**")
                    text = jd[:1000]
                    if len(jd) > 1000:
                        text += "\n\n... [truncated]"
                    st.markdown(text or "_None_")
                    st.markdown("**📰 News**")
                    text = raw_news[:1000]
                    if len(raw_news) > 1000:
                        text += "\n\n... [truncated]"
                    st.markdown(text or "_None_")
                    st.markdown("**📄 Papers**")
                    text = raw_papers[:1000]
                    if len(raw_papers) > 1000:
                        text += "\n\n... [truncated]"
                    st.markdown(text or "_None_")

        # --- Citations & sources expander (collapsed by default) ---
        with st.expander("📚 All Citations & Sources", expanded=False):
            for company in ranked:
                st.markdown(f"**{company}**")
                nd = news_data.get(company) or {}
                papers_text = nd.get("papers", "") or ""
                link = f"https://arxiv.org/search/?query={company}&searchtype=all"
                if papers_text.strip():
                    for line in papers_text.split("\n"):
                        line = line.strip()
                        if line:
                            st.markdown(f"- {line} — [Search arXiv]({link})")
                else:
                    st.markdown("_No arXiv papers listed._")

                st.markdown("**💼 Job Postings Sampled**")
                jd = (job_data.get(company) or {}).get("postings", "") or ""
                lines = [ln.strip() for ln in jd.split("\n") if ln.strip()][:8]
                for line in lines:
                    st.markdown(f"- {line}")
                if not lines:
                    st.markdown("_None_")

                st.markdown("**📰 News Signals**")
                raw_news = nd.get("news", "") or ""
                sentences = [s.strip() for s in raw_news.split(". ") if s.strip()][:3]
                for s in sentences:
                    st.markdown(f"- {s}.")
                if not sentences:
                    st.markdown("_None_")
                st.markdown("---")

    else:
        st.markdown(
            '<p style="color:#64748b;text-align:center;margin-top:2rem;">Select companies and click <strong>Run analysis</strong> to generate an intelligence brief.</p>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
