# IBM Telco Customer Churn — ML + FastAPI

Projeto didático end-to-end de previsão de churn em telecom.

## Dataset

Arquivo utilizado: `data/raw/Telco_customer_churn.xlsx`

Fonte de referência:
https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset

## Estrutura do projeto

- `notebooks/01_telco_customer_churn_ibm_end_to_end.ipynb`: notebook principal
- `data/raw/`: base original
- `src/train.py`: treinamento reproduzível
- `artifacts/`: modelo `.joblib` e metadados
- `app/main.py`: API FastAPI
- `docs/`: documentação da base, API, deploy e validação
- `tests/`: testes automatizados

## Executar localmente

```bash
pip install -r requirements.txt
python -m src.train
python -m uvicorn app.main:app --reload
```

Interface:
`http://127.0.0.1:8000`

Swagger:
`http://127.0.0.1:8000/docs`
