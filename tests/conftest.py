"""Test fixtures.

The tests target the CHECKPOINT-4 app (the final state). pyproject.toml
puts every checkpoint's directory on the pythonpath; tests import from
`app.*` and the checkpoint-4 layout is what wins (it has every module).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make sure CP4 is the `app` package the test session sees.
# Inserting at index 0 guarantees CP4's ``app`` package wins over any
# earlier checkpoint's partial implementation that pyproject.toml may
# have placed on ``sys.path``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "checkpoints" / "checkpoint-4-orchestration"))

# Provide a fake key so `Settings()` validates without a real provider.
# ``setdefault`` means a developer can still export a real key locally
# (e.g. for live integration testing) and it won't be clobbered.
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake")
