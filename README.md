# IBM Telco Customer Churn — ML + FastAPI

Projeto didático end-to-end de previsão de churn em telecom.

## Dataset
Arquivo incluído: `data/raw/Telco_customer_churn.xlsx`  
Referência: https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset

## Estrutura
- `notebooks/01_telco_customer_churn_ibm_end_to_end.ipynb`: notebook principal
- `data/raw/`: Excel original fornecido
- `src/train.py`: treinamento reproduzível do modelo de deploy
- `artifacts/`: modelo `.joblib` e metadados
- `app/main.py`: FastAPI
- `docs/`: dataset, API, deploy e validação
- `tests/`: testes automatizados

## Executar
```bash
pip install -r requirements.txt
python -m src.train
uvicorn app.main:app --reload
```

Swagger: `http://127.0.0.1:8000/docs`
