from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from src.modeling import risk_band

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "artifacts" / "telecom_churn_model.joblib"
artifact = joblib.load(ARTIFACT_PATH) if ARTIFACT_PATH.exists() else None

app = FastAPI(
    title="IBM Telco Customer Churn API",
    version="1.0.0",
    description="API didática para estimar probabilidade de churn em telecom.",
)

YesNo = Literal["Yes", "No"]


class CustomerInput(BaseModel):
    Gender: Literal["Male", "Female"] = "Female"
    Senior_Citizen: YesNo = Field("No", alias="Senior Citizen")
    Partner: YesNo = "No"
    Dependents: YesNo = "No"
    Tenure_Months: int = Field(12, ge=0, alias="Tenure Months")
    Phone_Service: YesNo = Field("Yes", alias="Phone Service")
    Multiple_Lines: str = Field("No", alias="Multiple Lines")
    Internet_Service: str = Field("Fiber optic", alias="Internet Service")
    Online_Security: str = Field("No", alias="Online Security")
    Online_Backup: str = Field("No", alias="Online Backup")
    Device_Protection: str = Field("No", alias="Device Protection")
    Tech_Support: str = Field("No", alias="Tech Support")
    Streaming_TV: str = Field("No", alias="Streaming TV")
    Streaming_Movies: str = Field("No", alias="Streaming Movies")
    Contract: str = "Month-to-month"
    Paperless_Billing: YesNo = Field("Yes", alias="Paperless Billing")
    Payment_Method: str = Field("Electronic check", alias="Payment Method")
    Monthly_Charges: float = Field(75.0, ge=0, alias="Monthly Charges")
    Total_Charges: float | None = Field(900.0, ge=0, alias="Total Charges")

    model_config = {"populate_by_name": True}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": artifact is not None}


@app.post("/predict")
def predict(customer: CustomerInput):
    if artifact is None:
        return {"error": "Modelo não encontrado. Execute: python -m src.train"}

    row = pd.DataFrame([customer.model_dump(by_alias=True)])
    probability = float(artifact["pipeline"].predict_proba(row)[0, 1])
    cutoff = float(artifact["cutoff"])
    prediction = int(probability >= cutoff)
    grade, risk_class, band = risk_band(probability)

    return {
        "churn_probability": round(probability, 6),
        "churn_probability_pct": round(probability * 100, 2),
        "prediction": prediction,
        "prediction_label": "Churn" if prediction else "Não Churn",
        "cutoff": round(cutoff, 6),
        "risk_grade": grade,
        "risk_class": risk_class,
        "risk_band": band,
        "model_name": artifact["model_name"],
        "model_version": artifact["model_version"],
    }


@app.get("/", response_class=HTMLResponse)
def home():
    # Interface simples de negócio. O objetivo não é substituir um front-end
    # corporativo, mas mostrar que a API pode ser consumida sem notebook.
    return r'''<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Telco Churn</title>
<style>
body{font-family:Arial,sans-serif;background:#0f172a;color:#e2e8f0;margin:0}.wrap{max-width:1050px;margin:auto;padding:30px}
h1{margin-bottom:4px}.sub{color:#94a3b8}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-top:22px}
label{font-size:13px;color:#cbd5e1}input,select{width:100%;box-sizing:border-box;padding:9px;margin-top:4px;background:#1e293b;color:#fff;border:1px solid #475569;border-radius:7px}
button{margin-top:20px;padding:12px 22px;border:0;border-radius:8px;font-weight:bold;cursor:pointer}.result{margin-top:24px;background:#1e293b;padding:20px;border-radius:12px;white-space:pre-wrap}a{color:#60a5fa}
</style></head><body><div class="wrap"><h1>Customer Churn — Telecom</h1><div class="sub">Interface didática consumindo a mesma API disponível em <a href="/docs">/docs</a>.</div>
<form id="f"><div class="grid">
<label>Gender<select name="Gender"><option>Female</option><option>Male</option></select></label>
<label>Senior Citizen<select name="Senior Citizen"><option>No</option><option>Yes</option></select></label>
<label>Partner<select name="Partner"><option>No</option><option>Yes</option></select></label>
<label>Dependents<select name="Dependents"><option>No</option><option>Yes</option></select></label>
<label>Tenure Months<input name="Tenure Months" type="number" value="12"></label>
<label>Phone Service<select name="Phone Service"><option>Yes</option><option>No</option></select></label>
<label>Multiple Lines<select name="Multiple Lines"><option>No</option><option>Yes</option><option>No phone service</option></select></label>
<label>Internet Service<select name="Internet Service"><option>Fiber optic</option><option>DSL</option><option>No</option></select></label>
<label>Online Security<select name="Online Security"><option>No</option><option>Yes</option><option>No internet service</option></select></label>
<label>Online Backup<select name="Online Backup"><option>No</option><option>Yes</option><option>No internet service</option></select></label>
<label>Device Protection<select name="Device Protection"><option>No</option><option>Yes</option><option>No internet service</option></select></label>
<label>Tech Support<select name="Tech Support"><option>No</option><option>Yes</option><option>No internet service</option></select></label>
<label>Streaming TV<select name="Streaming TV"><option>No</option><option>Yes</option><option>No internet service</option></select></label>
<label>Streaming Movies<select name="Streaming Movies"><option>No</option><option>Yes</option><option>No internet service</option></select></label>
<label>Contract<select name="Contract"><option>Month-to-month</option><option>One year</option><option>Two year</option></select></label>
<label>Paperless Billing<select name="Paperless Billing"><option>Yes</option><option>No</option></select></label>
<label>Payment Method<select name="Payment Method"><option>Electronic check</option><option>Mailed check</option><option>Bank transfer (automatic)</option><option>Credit card (automatic)</option></select></label>
<label>Monthly Charges<input name="Monthly Charges" type="number" step="0.01" value="75"></label>
<label>Total Charges<input name="Total Charges" type="number" step="0.01" value="900"></label>
</div><button type="submit">Calcular risco de churn</button></form><div id="r" class="result">Preencha os dados e clique em calcular.</div>
<script>
document.getElementById('f').onsubmit=async(e)=>{e.preventDefault();const fd=new FormData(e.target);const x=Object.fromEntries(fd.entries());
['Tenure Months','Monthly Charges','Total Charges'].forEach(k=>x[k]=Number(x[k]));
const res=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(x)});const j=await res.json();
document.getElementById('r').textContent=JSON.stringify(j,null,2)};
</script></div></body></html>'''
