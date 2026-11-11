from pathlib import Path
import yaml


def _find_default() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "configs" / "default.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("configs/default.yaml not found")


def load_config(path=None) -> dict:
    """Load the YAML config. Defaults to configs/default.yaml."""
    p = Path(path) if path else _find_default()
    with open(p) as f:
        return yaml.safe_load(f)
