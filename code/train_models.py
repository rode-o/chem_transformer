import os
import numpy as np
import joblib
import h5py
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
synth_data_h5_dir = os.path.join(project_root, "synth_data_h5")
model_dir = os.path.join(project_root, "model")
os.makedirs(model_dir, exist_ok=True)

h5_path = os.path.join(synth_data_h5_dir, "synthetic_dataset.h5")

with h5py.File(h5_path, 'r') as f:
    chemicals = f["Chemical"][:]            
    concentrations = f["Concentration"][:]
    experiments = f["Experiment_Number"][:]
    features = f["Features"][:]
    all_chemical_names = f["All_Chemical_Names"][:].astype(str)

# Fit LabelEncoder on the full set of chemical names
le = LabelEncoder()
le.fit(all_chemical_names)

# Convert numeric chemical IDs to string names
chemical_labels = np.array(all_chemical_names)[chemicals]
y_chemical = le.transform(chemical_labels)

# Aggregate by (y_chemical, concentrations, experiments)
int_concentrations = concentrations.astype(int)
combined_keys = np.stack([y_chemical, int_concentrations, experiments], axis=1)
sorted_idx = np.lexsort((experiments, int_concentrations, y_chemical))
sorted_keys = combined_keys[sorted_idx]
sorted_features = features[sorted_idx]

unique_keys, unique_idx, unique_counts = np.unique(sorted_keys, axis=0, return_index=True, return_counts=True)
num_groups = len(unique_keys)
agg_features = np.zeros((num_groups, features.shape[1]), dtype=np.float32)

start = 0
for i, count in enumerate(unique_counts):
    end = start + count
    agg_features[i] = sorted_features[start:end].mean(axis=0)
    start = end

group_chemicals = unique_keys[:, 0]
group_concentrations = unique_keys[:, 1].astype(np.float32)

X = agg_features
y_chemical_enc = group_chemicals
y_concentration = group_concentrations

# Scale Features
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

# Scale Concentration Targets
conc_scaler = StandardScaler()
X_train, X_test, y_chem_train, y_chem_test, y_conc_train_raw, y_conc_test_raw = train_test_split(
    X_scaled, y_chemical_enc, y_concentration, test_size=0.2, random_state=42
)

y_conc_train_scaled = conc_scaler.fit_transform(y_conc_train_raw.reshape(-1, 1)).ravel()
y_conc_test_scaled = conc_scaler.transform(y_conc_test_raw.reshape(-1, 1)).ravel()

# Expanded parameter grids for finer tuning
param_grid_classifier = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.05, 0.01],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 2]
}

param_grid_regressor = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.05, 0.01],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 2]
}

# Chemical Classification
chem_model_base = XGBClassifier(eval_metric='mlogloss', random_state=42, use_label_encoder=False)
chem_grid = GridSearchCV(
    chem_model_base,
    param_grid_classifier,
    cv=3,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)
chem_grid.fit(X_train, y_chem_train)
best_chem_model = chem_grid.best_estimator_

y_chem_pred = best_chem_model.predict(X_test)
chem_acc = accuracy_score(y_chem_test, y_chem_pred)
print("Best parameters (chemical):", chem_grid.best_params_)
print("Chemical Accuracy:", chem_acc)
print("Confusion Matrix:\n", confusion_matrix(y_chem_test, y_chem_pred))

# Concentration Regression
conc_model_base = XGBRegressor(random_state=42)
conc_grid = GridSearchCV(
    conc_model_base,
    param_grid_regressor,
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1
)
conc_grid.fit(X_train, y_conc_train_scaled)
best_conc_model = conc_grid.best_estimator_

y_conc_pred_scaled = best_conc_model.predict(X_test)
y_conc_pred = conc_scaler.inverse_transform(y_conc_pred_scaled.reshape(-1, 1)).ravel()
mse = mean_squared_error(y_conc_test_raw, y_conc_pred)
rmse = np.sqrt(mse)
print("Best parameters (concentration):", conc_grid.best_params_)
print("Concentration RMSE:", rmse)

chem_model_path = os.path.join(model_dir, "chemical_model.json")
conc_model_path = os.path.join(model_dir, "concentration_model.json")
feature_scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
conc_scaler_path = os.path.join(model_dir, "conc_scaler.pkl")
encoder_path = os.path.join(model_dir, "chemical_encoder.pkl")

best_chem_model.save_model(chem_model_path)
best_conc_model.save_model(conc_model_path)
joblib.dump(feature_scaler, feature_scaler_path)
joblib.dump(conc_scaler, conc_scaler_path)
joblib.dump(le, encoder_path)

print("\nModels, scalers, and encoder saved to:", model_dir)
print("Try adjusting param_grid and consider RandomizedSearchCV or Bayesian optimization for further improvement.")
