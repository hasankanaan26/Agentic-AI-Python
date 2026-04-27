"""Agent implementations used by the HTTP routes.

This package contains two parallel implementations of the same agent:

- ``raw_loop`` — a from-scratch async think/act/observe loop, kept around so
  engineers can read the simplest possible agent before opening LangGraph.
- ``langgraph`` — the production agent built on ``langgraph.prebuilt.create_react_agent``
  with checkpointing and (optional) human-in-the-loop interrupts.
"""
