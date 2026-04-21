import os
import logging
from typing import Literal, Optional, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from utils.config_loader import load_config
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

load_dotenv()
logger = logging.getLogger(__name__)


class _ConfigLoader:
    """Internal config wrapper so ModelLoader can stay a Pydantic BaseModel."""
    def __init__(self):
        self.config = load_config()
        logger.info("Config loaded successfully.")

    def __getitem__(self, key):
        return self.config[key]


class ModelLoader(BaseModel):
    model_provider: Literal["groq", "openai"] = "groq"
    config: Optional[_ConfigLoader] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        # Pydantic v2 post-init hook — runs after __init__
        self.config = _ConfigLoader()

    def load_llm(self):
        """
        Load and return the configured LLM.

        BUG FIX (original): OpenAI branch previously hardcoded 'o4-mini',
        ignoring whatever was set in config.yaml. Both branches now respect config.
        """
        logger.info(f"Loading LLM from provider: {self.model_provider}")

        if self.model_provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise EnvironmentError("GROQ_API_KEY not set in environment.")
            model_name = self.config["llm"]["groq"]["model_name"]
            logger.info(f"Using Groq model: {model_name}")
            return ChatGroq(model=model_name, api_key=api_key)

        elif self.model_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY not set in environment.")
            # BUG FIX: was hardcoded as "o4-mini" — now reads from config
            model_name = self.config["llm"]["openai"]["model_name"]
            logger.info(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(model=model_name, api_key=api_key)

        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")
