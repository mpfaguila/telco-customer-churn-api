# API

Execute `python -m src.train` e depois `uvicorn app.main:app --reload`.

- `GET /health` — saúde do serviço
- `POST /predict` — previsão de churn
- `GET /docs` — Swagger
- `GET /redoc` — ReDoc

Exemplo de payload em `examples/sample_request.json`.
