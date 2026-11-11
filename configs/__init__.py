"""Configuration loader."""
import yaml
from pathlib import Path
from typing import Dict

CONFIG_PATH = Path(__file__).parent / "default.yaml"

def load_config(config_path: Optional[Path] = None) -> Dict:
    """Load YAML configuration."""
    path = config_path or CONFIG_PATH
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config