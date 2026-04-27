#!/usr/bin/env python
"""Launch any checkpoint (or the base-app) on http://localhost:8000.

    python run_checkpoint.py base-app
    python run_checkpoint.py checkpoint-1-tool-calling
    python run_checkpoint.py checkpoint-2-agent-loop
    python run_checkpoint.py checkpoint-3-safety-rag
    python run_checkpoint.py checkpoint-4-orchestration

Pass --reload during live coding to enable autoreload.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

# Resolve the directory this script lives in so the launcher works no
# matter what the caller's cwd happens to be.
PROJECT_ROOT = Path(__file__).resolve().parent

# Map of CLI target name -> directory containing that target's ``app``
# package. New checkpoints just need a new entry here.
KNOWN_TARGETS = {
    "base-app": PROJECT_ROOT / "base-app",
    "checkpoint-1-tool-calling": PROJECT_ROOT / "checkpoints" / "checkpoint-1-tool-calling",
    "checkpoint-2-agent-loop": PROJECT_ROOT / "checkpoints" / "checkpoint-2-agent-loop",
    "checkpoint-3-safety-rag": PROJECT_ROOT / "checkpoints" / "checkpoint-3-safety-rag",
    "checkpoint-4-orchestration": PROJECT_ROOT / "checkpoints" / "checkpoint-4-orchestration",
}


def main() -> int:
    """Parse CLI args and start uvicorn for the requested checkpoint.

    Returns:
        Process exit code: ``0`` on a normal uvicorn shutdown, ``1``
        if the requested target's ``app/main.py`` is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    # ``choices=`` gives argparse-level validation + a helpful usage
    # message listing every supported target.
    parser.add_argument("target", choices=sorted(KNOWN_TARGETS.keys()))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="autoreload (live coding)")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args()

    target_dir = KNOWN_TARGETS[args.target]
    # Catch the most common breakage (a checkpoint that hasn't been
    # implemented yet) up front, before uvicorn produces a less obvious
    # ImportError traceback.
    if not (target_dir / "app" / "main.py").exists():
        print(f"error: missing {target_dir}/app/main.py", file=sys.stderr)
        return 1

    # `--app-dir` puts <target>/app on sys.path so `app.main:app` resolves.
    # We also insert it into the parent process's sys.path so anything
    # the launcher imports later (e.g. during reload coordination)
    # resolves against the right checkpoint.
    sys.path.insert(0, str(target_dir))

    print(f"-> serving {args.target} on http://{args.host}:{args.port}")
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        # Only watch the active checkpoint when --reload is on; otherwise
        # uvicorn would default to the cwd and pick up unrelated files.
        reload_dirs=[str(target_dir / "app")] if args.reload else None,
        log_level=args.log_level,
        app_dir=str(target_dir),
    )
    return 0


if __name__ == "__main__":
    # ``raise SystemExit`` propagates the int return code to the shell
    # so CI / shell wrappers can react to a missing-target failure.
    raise SystemExit(main())
