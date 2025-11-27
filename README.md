# AgriSense — Autonomous Crop Advisory & Disease Prediction Agent System

## Overview
AgriSense is a multi-agent system built with ADK-style patterns (as taught in Kaggle's 5-Day Agents Intensive).
It provides image-based crop diagnosis, actionable advisory, market insights, and climate risk checks.

## Features
- Multi-backend diagnosis (Local / MCP / Gemini)
- LLM-based advisory (Gemini) with template fallback
- Market & weather tools (open placeholders)
- Session & Memory Bank (InMemory)
- Observability (structured logging)
- Long-running climate monitor (threaded)

## Getting started (Kaggle)
1. Upload this repo to Kaggle or mount via GitHub.
2. Add any API keys to Kaggle Secrets:
   - `GEMINI_API_KEY` (if using Gemini)
   - `MCP_GATEWAY_URL` (if running MCP server)
   - `WEATHER_API_KEY` (optional)
3. Replace `NotImplementedError` placeholders in:
   - `agents/diagnosis_agent.py::_run_local`
   - `agents/diagnosis_agent.py::_run_gemini`
   - `agents/advisory_agent.py::_call_gemini`
   with the exact code cells from your 5-Day notebooks (model loading and Gemini calls).
4. Run the `submission_notebook.ipynb` and demo: upload image -> run orchestrator -> view advice.

## Kaggle Submission Checklist
- [ ] Notebook demonstrates multi-agent flows (diagnosis, advisory, market, climate)
- [ ] README explains architecture and setup
- [ ] Code comments explaining where to insert keys & model code
- [ ] No API keys in code

## Notes for integration
- Use the ADK MCP server (npx @modelcontextprotocol/server-everything) if you want to use MCP tools.
- For Gemini Vision, use the exact client instantiation you used in the 5-day notebooks, and set `gemini_client` in Orchestrator.
