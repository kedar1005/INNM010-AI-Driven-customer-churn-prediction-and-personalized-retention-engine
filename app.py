# app.py
import os, json, joblib, subprocess, threading
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd

from data_prep import load_data, preprocess
from retention import recommend_actions

CONFIG = json.load(open('sample_config.json'))
app = FastAPI(title="Churn Prediction & Retention API")

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_ART = None
COLUMNS = None

# ------------ LOAD MODEL ------------
def load_model_artifact():
    global MODEL_ART, COLUMNS
    path = os.path.join(CONFIG['model_dir'], CONFIG['model_file'])
    if os.path.exists(path):
        MODEL_ART = joblib.load(path)
        COLUMNS = MODEL_ART.get('columns', [])
        print("Model loaded with", len(COLUMNS), "columns")
    else:
        print("Model artifact not found:", path)

load_model_artifact()


# ------------ INDEX PAGE ------------
@app.get("/", response_class=HTMLResponse)
def index():
    if os.path.exists("templates/index.html"):
        with open("templates/index.html","r",encoding="utf8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h3>Dashboard not found. Put templates/index.html</h3>")


# ------------ METRICS ------------
@app.get("/metrics")
def metrics():
    df = load_data(CONFIG['data_path'])
    churn_col = next((c for c in df.columns if c.lower().startswith('churn')), None)

    total = len(df)
    churned = 0
    if churn_col:
        churned = df[df[churn_col].isin([1,'Yes',True])].shape[0]

    rev = df['MonthlyCharges'].sum() if 'MonthlyCharges' in df.columns else None

    return {
        "total_customers": total,
        "churned_customers": churned,
        "churn_rate": churned/total if total else 0,
        "monthly_revenue": rev
    }


# ------------ HIGH RISK API (with 4 categories) ------------
@app.get("/customers/highrisk")
def highrisk(limit:int=50):
    if MODEL_ART is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = load_data(CONFIG['data_path'])
    df_proc = preprocess(df)

    target_col = next((c for c in df_proc.columns if c.lower().startswith('churn')), None)
    X = df_proc.drop(columns=[target_col]) if target_col else df_proc

    # Ensure all training columns exist
    for c in COLUMNS:
        if c not in X.columns:
            X[c] = 0

    X = X[COLUMNS]
    probs = MODEL_ART['model'].predict_proba(X)[:,1]

    df2 = df.copy()
    df2['churn_prob'] = probs
    df2 = df2.sort_values('churn_prob', ascending=False).head(limit)

    rows = []

    for _, r in df2.iterrows():
        prob = float(r['churn_prob'])
        prob_pct = round(prob * 100)

        # 4-category risk bucket
        if prob_pct >= 85:
            risk = "Critical"
        elif prob_pct >= 60:
            risk = "High"
        elif prob_pct >= 30:
            risk = "Mid"
        else:
            risk = "Low"

        rows.append({
            "customerID": r.get("customerID", None),
            "churn_prob": prob_pct,
            "risk": risk
        })

    return {"rows": rows}


# ------------ PREDICT API ------------
@app.post("/predict")
async def predict(records: list):
    if not records:
        raise HTTPException(status_code=400, detail="No records provided")
    if MODEL_ART is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame(records)

    for c in COLUMNS:
        if c not in df.columns:
            df[c] = 0

    X = df[COLUMNS]
    model = MODEL_ART['model']

    preds = model.predict(X).tolist()
    probs = model.predict_proba(X)[:,1].tolist()

    out = []

    shap_art_path = os.path.join(CONFIG['model_dir'],'shap_explainer.pkl')
    shap_art = joblib.load(shap_art_path) if os.path.exists(shap_art_path) else None

    for i,row in X.iterrows():
        prob = float(probs[i])

        # SHAP top features
        shap_top=[]
        if shap_art:
            expl = shap_art['explainer']
            cols = shap_art['cols']
            sv = expl(X.iloc[[i]])
            arr = sv.values[0]
            idxs = abs(arr).argsort()[-3:][::-1]
            shap_top = [cols[j] for j in idxs]

        actions = recommend_actions(row.to_dict(), prob, shap_top)

        out.append({
            "prediction": int(preds[i]),
            "probability": int(round(prob*100)),   # integer %
            "actions": actions,
            "top_shap": shap_top
        })

    return {"results": out}


# ------------ BACKGROUND RETRAIN ------------
def background_retrain():
    try:
        subprocess.call(["python", "train_model.py"])
        subprocess.call(["python", "explain_shap.py"])
        load_model_artifact()
        print("Retraining finished.")
    except Exception as e:
        print("Retrain error:", e)


@app.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...), retrain: bool = True):
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    dest = os.path.join(data_dir, "telco_customer_churn.csv")
    contents = await file.read()
    with open(dest, "wb") as f:
        f.write(contents)

    msg = "Dataset uploaded."

    if retrain:
        thread = threading.Thread(target=background_retrain)
        thread.start()
        msg += " Retraining started in background."

    return {"message": msg}


# ------------ ARTIFACTS ------------
@app.get("/artifacts/{name}")
def artifacts(name: str):
    path = os.path.join(CONFIG['artifact_dir'], name)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Artifact not found")
