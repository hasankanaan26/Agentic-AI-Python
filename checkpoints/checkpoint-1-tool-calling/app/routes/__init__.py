"""HTTP route modules grouped by domain.

Each submodule defines an `APIRouter` that `app.main` mounts on the FastAPI
application. Keeping routes split by domain (health, tools, ...) makes it
easy to test them in isolation and to reason about which dependencies each
endpoint actually needs.
"""
