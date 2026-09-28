from pathlib import Path
from src.modeling import load_data, FEATURES, TARGET
ROOT=Path(__file__).resolve().parents[1]
def test_dataset_schema():
    df=load_data(ROOT/'data/raw/Telco_customer_churn.xlsx')
    assert len(df)==7043
    assert TARGET in df.columns
    assert all(c in df.columns for c in FEATURES)
    assert set(df[TARGET].dropna().unique()).issubset({0,1})
