import os
import yaml

# Project root (parent of utils/) — so config is found regardless of CWD
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "config.yaml")


def load_config(config_path: str | None = None) -> dict:
    """Load YAML config file and return as dict."""
    path = config_path or _DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config
