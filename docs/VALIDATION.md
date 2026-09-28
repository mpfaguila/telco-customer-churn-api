# Validação técnica

A versão final foi validada com o arquivo real `Telco_customer_churn.xlsx` fornecido no projeto.

Foram verificados:

- leitura do Excel com 7.043 registros e 33 colunas;
- schema das 19 features de modelagem e target `Churn Value`;
- missing values e duplicidades;
- execução sequencial de todas as células de código do notebook;
- treinamento dos 5 modelos;
- Grid Search com 3-fold CV;
- cutoff OOF por máximo KS/Youden;
- SHAP;
- persistência e recarga do `.joblib`;
- script `python -m src.train`;
- API FastAPI, `/health` e `/predict`;
- testes automatizados com pytest.
