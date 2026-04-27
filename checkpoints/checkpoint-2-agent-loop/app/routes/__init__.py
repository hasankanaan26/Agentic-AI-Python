"""HTTP route handlers.

One module per logical resource:

* :mod:`app.routes.health` — liveness check.
* :mod:`app.routes.tools`  — single-shot tool calling and tool listing.
* :mod:`app.routes.agent`  — multi-step from-scratch agent loop.

Each module exposes a ``router`` :class:`fastapi.APIRouter` that
:mod:`app.main` mounts on the application.
"""
