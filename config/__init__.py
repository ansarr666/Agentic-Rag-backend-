import os
import yaml
from pathlib import Path
from typing import Any, Dict

CONFIG_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CONFIG_DIR.parent.resolve()
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"

def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration from YAML file and resolve relative paths to project root."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Automatically load .env if present
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            with open(env_file, "r", encoding="utf-8") as ef:
                for line in ef:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v

    return config

__all__ = ["load_config", "PROJECT_ROOT", "CONFIG_DIR"]
