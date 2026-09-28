from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn Value"

# Colunas usadas pelo modelo. Excluímos identificadores, geografia e campos que
# carregam informação posterior/derivada do churn para reduzir leakage.
FEATURES = [
    "Gender", "Senior Citizen", "Partner", "Dependents", "Tenure Months",
    "Phone Service", "Multiple Lines", "Internet Service", "Online Security",
    "Online Backup", "Device Protection", "Tech Support", "Streaming TV",
    "Streaming Movies", "Contract", "Paperless Billing", "Payment Method",
    "Monthly Charges", "Total Charges"
]

CATEGORICAL_FEATURES = [
    "Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service",
    "Multiple Lines", "Internet Service", "Online Security", "Online Backup",
    "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies",
    "Contract", "Paperless Billing", "Payment Method"
]
NUMERIC_FEATURES = ["Tenure Months", "Monthly Charges", "Total Charges"]

DROP_FROM_MODEL = [
    "CustomerID", "Count", "Country", "State", "City", "Zip Code", "Lat Long",
    "Latitude", "Longitude", "Churn Label", "Churn Score", "CLTV", "Churn Reason"
]

def load_data(path):
    df = pd.read_excel(path)
    # Total Charges vem como object porque há algumas células vazias/strings.
    # Convertemos com errors='coerce': entradas não numéricas tornam-se NaN e
    # serão tratadas pelo imputador dentro do pipeline, sem apagar clientes.
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    return df

def build_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ], remainder="drop", verbose_feature_names_out=False)

def risk_band(prob):
    if prob < 0.10: return "A", "Risco Muito Baixo", "< 10%"
    if prob < 0.25: return "B", "Risco Baixo", "10% a < 25%"
    if prob < 0.50: return "C", "Risco Médio", "25% a < 50%"
    if prob < 0.75: return "D", "Risco Alto", "50% a < 75%"
    return "E", "Risco Muito Alto", ">= 75%"
