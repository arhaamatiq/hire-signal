from typing import List, Dict

# Canonical AI domain taxonomy
# Maps normalized skill/keyword → domain category
SKILL_DOMAIN_MAP: Dict[str, str] = {
    # Inference & serving
    "triton": "inference optimization",
    "tensorrt": "inference optimization",
    "vllm": "inference optimization",
    "onnx": "inference optimization",
    "model serving": "inference optimization",
    "low latency inference": "inference optimization",
    "quantization": "inference optimization",
    "distillation": "inference optimization",

    # RAG & retrieval
    "rag": "rag",
    "retrieval augmented generation": "rag",
    "vector database": "rag",
    "embeddings": "rag",
    "semantic search": "rag",
    "weaviate": "rag",
    "pinecone": "rag",
    "chroma": "rag",
    "faiss": "rag",

    # Fine-tuning & post-training
    "fine-tuning": "fine-tuning",
    "finetuning": "fine-tuning",
    "lora": "fine-tuning",
    "qlora": "fine-tuning",
    "sft": "fine-tuning",
    "instruction tuning": "fine-tuning",
    "peft": "fine-tuning",

    # RLHF & Alignment
    "rlhf": "alignment",
    "reinforcement learning from human feedback": "alignment",
    "constitutional ai": "alignment",
    "dpo": "alignment",
    "reward model": "alignment",
    "alignment": "alignment",
    "red teaming": "alignment",

    # Foundation models
    "pretraining": "foundation models",
    "pre-training": "foundation models",
    "transformer": "foundation models",
    "large language model": "foundation models",
    "llm": "foundation models",
    "gpt": "foundation models",
    "llama": "foundation models",
    "foundation model": "foundation models",

    # MLOps
    "mlflow": "mlops",
    "kubeflow": "mlops",
    "mlops": "mlops",
    "model registry": "mlops",
    "experiment tracking": "mlops",
    "data pipeline": "mlops",
    "feature store": "mlops",
    "wandb": "mlops",

    # Infrastructure
    "cuda": "gpu infrastructure",
    "gpu cluster": "gpu infrastructure",
    "distributed training": "gpu infrastructure",
    "pytorch distributed": "gpu infrastructure",
    "horovod": "gpu infrastructure",
    "ray": "gpu infrastructure",
    "kubernetes": "gpu infrastructure",

    # Agents & orchestration
    "langchain": "ai agents",
    "langgraph": "ai agents",
    "agentic": "ai agents",
    "tool use": "ai agents",
    "function calling": "ai agents",
    "multi-agent": "ai agents",

    # Multimodal
    "vision language model": "multimodal",
    "vlm": "multimodal",
    "image generation": "multimodal",
    "diffusion model": "multimodal",
    "multimodal": "multimodal",
    "speech": "multimodal",
    "audio": "multimodal",
}

# High-signal domains that suggest imminent product launch
HIGH_SIGNAL_DOMAINS = {
    "inference optimization",
    "rag",
    "fine-tuning",
    "alignment",
    "foundation models",
    "ai agents",
    "multimodal",
}

# Keywords that indicate an AI role in job postings (for volume scoring)
_AI_ROLE_KEYWORDS = ["engineer", "scientist", "researcher"]


def map_skills_to_domains(skills: List[str]) -> List[str]:
    """Map a list of raw skill strings to their canonical AI domains."""
    domains = set()
    for skill in skills:
        normalized = skill.lower().strip()
        for keyword, domain in SKILL_DOMAIN_MAP.items():
            if keyword in normalized:
                domains.add(domain)
    return sorted(domains)


def score_domain_signals(domains: List[str]) -> int:
    """
    Score how many high-signal AI domains are present.
    Used as a component in the overall intent score.
    Returns 0-50.
    """
    detected_high_signal = set(d.lower() for d in domains) & HIGH_SIGNAL_DOMAINS
    return min(50, len(detected_high_signal) * 8)


def compute_intent_score(
    postings_text: str,
    domains: List[str],
    has_funding_news: bool,
    has_recent_papers: bool,
) -> int:
    """
    Score 0–100 from signals available on the first run (no historical baseline).
    Base 30 + job volume (up to 25) + domain coverage (up to 25) + funding (10) + papers (10), capped at 100.
    """
    # Base score so companies with no signal still get a floor
    score = 30

    # AI job posting volume: more "engineer"/"scientist"/"researcher" in postings = stronger hiring signal
    text_lower = (postings_text or "").lower()
    hits = sum(1 for kw in _AI_ROLE_KEYWORDS if kw in text_lower)
    score += min(5, hits) * 5

    # High-signal domain coverage: domains like "alignment" or "foundation models" suggest product focus
    high_signal_count = len(set(d.lower() for d in domains) & HIGH_SIGNAL_DOMAINS)
    score += min(5, high_signal_count) * 5

    # Funding/partnership news: raises and deals often precede hiring surges
    if has_funding_news:
        score += 10

    # Recent arXiv papers: indicates active research investment
    if has_recent_papers:
        score += 10

    return max(0, min(100, int(score)))
