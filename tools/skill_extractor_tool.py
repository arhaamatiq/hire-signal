import json
import logging
from typing import List
from langchain_core.tools import tool
from utils.skill_taxonomy import map_skills_to_domains

logger = logging.getLogger(__name__)


class SkillExtractorTool:
    """Tool wrappers for AI skill extraction and domain categorization.

    Unlike other tool classes, this one contains its own logic rather than
    delegating to a utils/ class — the extraction is simple enough to live here.
    """

    def __init__(self):
        self.tool_list = self._setup_tools()

    def _setup_tools(self) -> List:

        @tool
        def extract_ai_skills(postings_text: str) -> str:
            """Extract AI/ML skills, technologies, and frameworks from raw job
            posting text. Returns a JSON string with keys: top_skills, domains,
            role_types, team_signals.
            """
            if not postings_text or len(postings_text.strip()) < 20:
                return json.dumps({
                    "top_skills": [],
                    "domains": [],
                    "role_types": [],
                    "team_signals": [],
                })

            known_skills = [
                "pytorch", "tensorflow", "jax", "cuda", "triton", "tensorrt",
                "vllm", "onnx", "transformers", "hugging face", "langchain",
                "langgraph", "rag", "vector database", "pinecone", "weaviate",
                "faiss", "lora", "qlora", "rlhf", "dpo", "fine-tuning",
                "pre-training", "llm", "foundation model", "mlflow", "kubeflow",
                "ray", "kubernetes", "distributed training", "quantization",
                "inference", "multimodal", "vision", "speech", "reward model",
                "alignment", "red teaming", "mlops", "feature store",
            ]

            text_lower = postings_text.lower()
            found_skills = [s for s in known_skills if s in text_lower]
            domains = map_skills_to_domains(found_skills)

            role_keywords = {
                "ML Engineer":         ["machine learning engineer", "ml engineer"],
                "Research Scientist":  ["research scientist", "ai researcher"],
                "LLM Engineer":        ["llm engineer", "language model"],
                "MLOps Engineer":      ["mlops", "ml platform", "ml infrastructure"],
                "Applied Scientist":   ["applied scientist", "applied researcher"],
                "Data Scientist":      ["data scientist"],
            }
            role_types = [
                role for role, keywords in role_keywords.items()
                if any(kw in text_lower for kw in keywords)
            ]

            signal_phrases = {
                "scaling inference":          ["scaling inference", "inference at scale"],
                "building RAG pipeline":      ["rag pipeline", "retrieval system"],
                "training foundation model":  ["pre-training", "training llm"],
                "fine-tuning at scale":       ["fine-tuning", "finetuning"],
                "building alignment systems": ["rlhf", "alignment", "reward model"],
                "multi-agent systems":        ["multi-agent", "agentic", "langgraph"],
                "multimodal product":         ["multimodal", "vision language", "vlm"],
            }
            team_signals = [
                signal for signal, phrases in signal_phrases.items()
                if any(p in text_lower for p in phrases)
            ]

            return json.dumps({
                "top_skills":   found_skills[:15],
                "domains":      domains,
                "role_types":   role_types,
                "team_signals": team_signals,
            })

        @tool
        def categorize_ai_domain(skills_json: str) -> str:
            """Given a JSON string of extracted skills, return the primary AI
            domain this company is investing in (e.g. "inference optimization",
            "rag", "foundation models").
            """
            try:
                skills_data = json.loads(skills_json)
                domains = skills_data.get("domains", [])
                if not domains:
                    return "undetermined"
                return domains[0]
            except Exception:
                return "undetermined"

        return [extract_ai_skills, categorize_ai_domain]
