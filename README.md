# utah-tourism-agent
A LLM based chat that recommends places to visit in Utah


## Running
- `docker compose up --build`
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Plan me a 3-day trip to Zion and Bryce Canyon with good beginner hikes.",
    "history": []
  }'
```