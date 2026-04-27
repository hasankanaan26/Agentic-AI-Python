"""Checkpoint-1 application package.

This package wires up the FastAPI service that demonstrates the tool-calling
primitive: an async LLM client, a registry of callable tools, and a small
HTTP surface for invoking them. It is intentionally minimal — later
checkpoints layer in RAG, multi-step agents, observability, and safety.
"""
