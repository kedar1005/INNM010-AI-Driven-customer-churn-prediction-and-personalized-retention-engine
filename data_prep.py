# data_prep.py
import pandas as pd
import numpy as np

def load_data(path):
    # Detect extension
    ext = path.lower().split('.')[-1]

    if ext == 'csv':
        df = pd.read_csv(path)

    elif ext in ['xlsx', 'xls']:
        df = pd.read_excel(path)

    else:
        raise ValueError("Unsupported file format. Please upload CSV or Excel file.")

    return df
def basic_clean(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce')
    df = df.replace({'No internet service': 'No', 'No phone service': 'No'})
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes':1,'No':0}).fillna(df.get('Churn'))
    return df

def feature_engineering(df):
    df = df.copy()
    if 'tenure' in df.columns:
        df['tenure_bucket'] = pd.cut(df['tenure'], bins=[-1,3,12,24,48,100],
                                     labels=['0-3','4-12','13-24','25-48','48+'])
    services = ['PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
                'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    svc = [c for c in services if c in df.columns]
    if svc:
        for c in svc:
            df[c] = df[c].replace({'Yes':1,'No':0})
        df['service_count'] = df[svc].sum(axis=1)
    if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
        df['monthly_per_tenure'] = df['MonthlyCharges'] / (df['tenure'].replace(0,1))
    return df

def preprocess(df, drop_customer_id=True):
    df = basic_clean(df)
    df = feature_engineering(df)
    num_cols = df.select_dtypes(include=['float64','int64']).columns.tolist()
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    bin_map = {'Yes':1,'No':0}
    for c in df.select_dtypes(include=['object']).columns:
        if c == 'customerID':
            continue
        uniques = set(df[c].dropna().unique())
        if uniques and uniques.issubset(set(bin_map.keys())):
            df[c] = df[c].map(bin_map)
    if drop_customer_id and 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    df = pd.get_dummies(df, drop_first=True)
    df = df.fillna(0)
    return df
def feature_engineering(df):
    df = df.copy()

    # Tenure buckets
    if 'tenure' in df.columns:
        df['tenure_bucket'] = pd.cut(
            df['tenure'],
            bins=[-1, 3, 12, 24, 48, 100],
            labels=['0-3', '4-12', '13-24', '25-48', '48+']
        )

    # Service columns
    services = [
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]

    svc = [c for c in services if c in df.columns]

    # Normalize all service columns
    for c in svc:
        df[c] = df[c].astype(str).str.strip()

        df[c] = df[c].replace({
            'Yes': 1,
            'No': 0,
            'No internet service': 0,
            'No phone service': 0
        })

        # Convert any leftover strings to 0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # Now sum safely
    if svc:
        df['service_count'] = df[svc].sum(axis=1)

    # MonthlyCharges per tenure
    if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
        df['monthly_per_tenure'] = df['MonthlyCharges'] / df['tenure'].replace(0, 1)

    return df


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/telco_customer_churn.csv')
    args = p.parse_args()
    df = load_data(args.data)
    print('Loaded', df.shape)
    df2 = preprocess(df)
    print('After preprocess', df2.shape)
