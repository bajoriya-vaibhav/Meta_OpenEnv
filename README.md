---
title: ChronoVeritas
emoji: 😻
colorFrom: green
colorTo: pink
sdk: docker
pinned: false
short_description: An RL environment for temporal fact-checking
---


# ChronoVeritas — Claim Lifecycle Verification Environment

An [OpenEnv](https://openenv.ai)-compliant reinforcement learning environment for temporal fact-checking. Agents trace how a factual claim mutates across a small document corpus, identify the mutation point, classify the mutation type, and reconstruct the provenance chain.

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn server:app --port 8000

# Health check
curl http://localhost:8000/health

# Reset to a specific task
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "EASY-001"}'

# Execute a search step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"type": "search", "payload": {"query": "budget transport council"}}'
```

### Running Inference

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export OPENAI_API_KEY=sk-...

python inference.py
```

### Docker

```bash
docker build -t chronoveritas .
docker run -p 7860:7860 chronoveritas
```

## API Endpoints

| Method | Path     | Description                  |
|--------|----------|------------------------------|
| GET    | /health  | Health check                 |
| GET    | /tasks   | List available tasks         |
| POST   | /reset   | Start a new episode          |
| POST   | /step    | Execute an action            |
| GET    | /state   | Get current observation      |

## Actions

| Action               | Step Cost | Token Cost      | Description                        |
|----------------------|-----------|------------------|------------------------------------|
| search               | 1         | 0               | BM25 keyword search over corpus    |
| fetch_doc            | 1         | content_len // 4 | Load full document text            |
| add_timeline_event   | 0         | 0               | Annotate timeline                  |
| flag_contradiction   | 0         | 0               | Mark two docs as contradictory     |
| set_mutation_point   | 0         | 0               | Declare mutation point (+partial)  |
| submit_verdict       | 0         | 0               | Final verdict (terminal)           |

## Tasks

- **EASY-001**: Budget distortion (15 steps max)
- **MED-001**: Drug recall omission (20 steps max)
- **HARD-001**: Research fabrication (30 steps max)

## Reward Function

Fully deterministic scoring with no LLM-as-judge:
- Verdict accuracy
- Mutation type classification
- Mutation point identification
- Provenance chain F1
- Timeline ordering (Kendall tau)
- Step efficiency bonus
- Hallucination penalty

## Environment Variables

| Variable        | Default                      | Description           |
|-----------------|------------------------------|-----------------------|
| API_BASE_URL    | https://api.openai.com/v1    | LLM API endpoint      |
| MODEL_NAME      | gpt-4o-mini                  | LLM model name        |
| OPENAI_API_KEY  | (required)                   | API key for LLM calls |
| ENV_BASE_URL    | http://localhost:8000        | Environment server URL |
