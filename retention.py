# retention.py
def recommend_actions(row: dict, prob: float, shap_top_features: list = None):
    actions = []
    if prob >= 0.85:
        actions.append({"action":"Immediate retention call", "priority":"high"})
        actions.append({"action":"Offer 30% discount for 3 months", "priority":"high"})
    elif prob >= 0.65:
        actions.append({"action":"Priority support + retention call", "priority":"high"})
        actions.append({"action":"Offer 20% discount for 2 months", "priority":"medium"})
    elif prob >= 0.45:
        actions.append({"action":"Email with personalized offer", "priority":"medium"})
    else:
        actions.append({"action":"Monitor & send NPS survey", "priority":"low"})

    if 'tenure' in row and row.get('tenure',0) <= 3:
        actions.insert(0, {"action":"Onboarding call + setup help", "priority":"high"})

    if 'MonthlyCharges' in row and row.get('MonthlyCharges',0) > 100:
        actions.append({"action":"Offer billing optimization", "priority":"medium"})

    if shap_top_features:
        if any('Payment' in s or 'payment' in s.lower() for s in shap_top_features):
            actions.append({"action":"Contact to fix billing/payment issue", "priority":"high"})

    seen = set(); out=[]
    for a in actions:
        if a['action'] not in seen:
            out.append(a); seen.add(a['action'])
    return out
