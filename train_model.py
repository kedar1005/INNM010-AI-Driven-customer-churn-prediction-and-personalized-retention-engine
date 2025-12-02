# train_model.py
import os, json, joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
import xgboost as xgb
from imblearn.over_sampling import SMOTE

from data_prep import load_data, preprocess
from imblearn.over_sampling import SMOTE



CONFIG = json.load(open('sample_config.json'))
os.makedirs(CONFIG['model_dir'], exist_ok=True)
os.makedirs(CONFIG['artifact_dir'], exist_ok=True)

def get_Xy(df_proc):
    target_cols = [c for c in df_proc.columns if c.lower().startswith('churn')]
    if not target_cols:
        raise ValueError("No target 'Churn' column found after preprocessing.")
    target = target_cols[0]
    X = df_proc.drop(columns=[target])
    y = df_proc[target].astype(int)
    return X, y

def plot_confusion(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(path); plt.close()

def plot_roc(y_true, y_score, path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], '--', color='gray')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve')
    plt.tight_layout()
    plt.savefig(path); plt.close()

def train():
    print("Loading dataset:", CONFIG['data_path'])
    df = load_data(CONFIG['data_path'])
    print("Preprocessing...")
    df_proc = preprocess(df)
    X, y = get_Xy(df_proc)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=CONFIG.get('test_size',0.2),
                                                        random_state=CONFIG['random_state'],
                                                        stratify=y)
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    print("Train churn ratio:", y_train.mean())
    if y_train.mean() < 0.3:
        sm = SMOTE(random_state=CONFIG['random_state'])
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        print("After SMOTE ratio:", y_train_res.mean())
    else:
        X_train_res, y_train_res = X_train, y_train

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                              random_state=CONFIG['random_state'], n_estimators=100)
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else None

    print("Accuracy:", accuracy_score(y_test, y_pred))
    if y_proba is not None:
        print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred))

    artifact = {'model': model, 'columns': X.columns.tolist()}
    model_path = os.path.join(CONFIG['model_dir'], CONFIG['model_file'])
    joblib.dump(artifact, model_path)
    print("Saved model to", model_path)

    # Save artifacts
    plot_confusion(y_test, y_pred, os.path.join(CONFIG['artifact_dir'], 'confusion_matrix.png'))
    if y_proba is not None:
        plot_roc(y_test, y_proba, os.path.join(CONFIG['artifact_dir'], 'roc_curve.png'))
    try:
        fi = model.feature_importances_
        feat = X.columns
        df_fi = pd.DataFrame({'feature': feat, 'importance': fi}).sort_values('importance', ascending=False).head(30)
        plt.figure(figsize=(8,6))
        sns.barplot(x='importance', y='feature', data=df_fi)
        plt.title('Top Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG['artifact_dir'], 'feature_importance.png'))
        plt.close()
    except Exception as e:
        print("Feature importance error:", e)

if __name__ == "__main__":
    train()
