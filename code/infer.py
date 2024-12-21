import os
import numpy as np
import h5py
import joblib
from xgboost import XGBClassifier, XGBRegressor

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(project_root, "model")
infer_data_dir = os.path.join(project_root, "infer_data")

chem_model_path = os.path.join(model_dir, "chemical_model.json")
conc_model_path = os.path.join(model_dir, "concentration_model.json")
feature_scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
conc_scaler_path = os.path.join(model_dir, "conc_scaler.pkl")
encoder_path = os.path.join(model_dir, "chemical_encoder.pkl")

chem_model = XGBClassifier()
chem_model.load_model(chem_model_path)

conc_model = XGBRegressor()
conc_model.load_model(conc_model_path)

feature_scaler = joblib.load(feature_scaler_path)
conc_scaler = joblib.load(conc_scaler_path)
le = joblib.load(encoder_path)

# Example: run inference on all .h5 files in infer_data
for file_name in os.listdir(infer_data_dir):
    if file_name.endswith(".h5"):
        infer_h5_path = os.path.join(infer_data_dir, file_name)
        with h5py.File(infer_h5_path, 'r') as f:
            features = f["Features"][:]

        # Aggregate frequency points (mean)
        X_infer = features.mean(axis=0, keepdims=True)

        # Apply feature scaling
        X_infer_scaled = feature_scaler.transform(X_infer)

        # Predict chemical
        y_chem_pred = chem_model.predict(X_infer_scaled)[0]
        predicted_chemical_name = le.inverse_transform([y_chem_pred])[0]

        # Predict concentration (scaled)
        y_conc_pred_scaled = conc_model.predict(X_infer_scaled)
        y_conc_pred = conc_scaler.inverse_transform(y_conc_pred_scaled.reshape(-1,1)).ravel()[0]

        print(f"Inference Results for {file_name}:")
        print("Predicted Chemical:", predicted_chemical_name)
        print("Predicted Concentration:", y_conc_pred)
        print("-" * 50)
