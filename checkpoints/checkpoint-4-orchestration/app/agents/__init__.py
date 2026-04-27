"""Agent implementations.

* :mod:`app.agents.raw_loop` -- the from-scratch async think/act/observe loop,
  kept around as the pedagogical baseline.
* :mod:`app.agents.langgraph` -- the LangGraph ReAct agent wired to our tools,
  with checkpointing, streaming, and human-in-the-loop interrupts.
* :mod:`app.agents.studio` -- a slim graph exposed to LangGraph Studio for
  visual debugging.
"""
