import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Tests must never touch a developer's real database or model directory.
os.environ.setdefault("DATABASE_URL", "sqlite:///./data/test.db")
os.environ.setdefault("MODEL_DIR", "./data/test-models")
os.environ.setdefault("OPERATOR_TOKEN", "test-token")
