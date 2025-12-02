
1)Train model (first time or after major data change):
   python train_model.py

2)Generate SHAP artifacts:
   python explain_shap.py

3)Run server:
   uvicorn app:app --reload --port 8000

4)Open dashboard:
   http://127.0.0.1:8000/

