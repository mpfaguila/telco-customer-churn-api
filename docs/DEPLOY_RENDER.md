# Deploy no Render

1. Publique o projeto no GitHub.
2. Crie um Web Service no Render.
3. Build command: `pip install -r requirements.txt && python -m src.train`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Após o deploy, acesse `/docs` para testar a API.
