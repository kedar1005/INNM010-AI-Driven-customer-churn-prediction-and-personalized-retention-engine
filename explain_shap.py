# explain_shap.py
import os, json, joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from data_prep import load_data, preprocess

CONFIG = json.load(open('sample_config.json'))
os.makedirs(CONFIG['artifact_dir'], exist_ok=True)

def save_shap_summary():
    artifact = joblib.load(os.path.join(CONFIG['model_dir'], CONFIG['model_file']))
    model = artifact['model']
    cols = artifact['columns']
    df = load_data(CONFIG['data_path'])
    df_proc = preprocess(df)
    target_col = [c for c in df_proc.columns if c.lower().startswith('churn')][0]
    X = df_proc.drop(columns=[target_col])[cols]
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    plt.figure(figsize=(8,6))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['artifact_dir'],'shap_summary.png'))
    plt.close()
    joblib.dump({'explainer': explainer, 'cols': cols}, os.path.join(CONFIG['model_dir'],'shap_explainer.pkl'))
    print("Saved SHAP summary and explainer.")

def save_shap_for(index=0):
    pe = joblib.load(os.path.join(CONFIG['model_dir'],'shap_explainer.pkl'))
    explainer = pe['explainer']
    cols = pe['cols']
    df = load_data(CONFIG['data_path'])
    df_proc = preprocess(df)
    target_col = [c for c in df_proc.columns if c.lower().startswith('churn')][0]
    X = df_proc.drop(columns=[target_col])[cols]
    sv = explainer(X.iloc[[index]])
    plt.figure(figsize=(6,4))
    shap.plots.waterfall(sv[0], show=False)
    plt.tight_layout()
    out = os.path.join(CONFIG['artifact_dir'], f'shap_customer_{index}.png')
    plt.savefig(out)
    plt.close()
    print("Saved per-customer SHAP:", out)

if __name__ == "__main__":
    save_shap_summary()
    save_shap_for(0)
