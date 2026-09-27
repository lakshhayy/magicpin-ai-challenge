# Magicpin AI Challenge - Team Antigravity

## Approach
This bot implements a resilient, state-machine driven routing layer for merchant and customer engagement. 

- **Intelligent Routing**: Determines context (merchant vs customer) and routes to specialized handlers.
- **Strict Adherence**: Uses system prompts strictly aligned with the 5-dimension scoring rubric (Specificity, Category Fit, Merchant Fit, Trigger Relevance, Engagement Compulsion).
- **Rate-Limit Resilience**: Implements robust retry logic (exponential backoff) to gracefully handle LLM API rate limits (e.g., HTTP 429 Too Many Requests).
- **FastAPI Backend**: Provides asynchronous endpoints (`/v1/healthz`, `/v1/metadata`, `/v1/context`, `/v1/tick`, `/v1/reply`) to interface seamlessly with the Magicpin Judge.

## How to Run
```bash
pip install -r requirements.txt
uvicorn bot:app --port 8080
```

## Team
- **Name**: Team Lakshay
- **LLM Used**: Groq API (openai/gpt-oss-120b)
