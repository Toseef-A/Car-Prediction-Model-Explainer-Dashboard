import pandas as pd
import joblib
from explainerdashboard import RegressionExplainer, ExplainerDashboard

# Load model and data
model = joblib.load("best_xgb_model.pkl")
X_test_scaled = joblib.load("X_test_scaled.pkl")
y_test = joblib.load("y_test.pkl")
encoders = joblib.load("label_encoders.pkl")
price_scaler = joblib.load("price_scaler.pkl")

# Reverse scale y_test
y_test_unscaled = price_scaler.inverse_transform(y_test.values.reshape(-1, 1)).flatten()

# Make a copy of X_test_scaled with decoded labels for human-friendly display
X_test_readable = X_test_scaled.copy()
for col, le in encoders.items():
    if col in X_test_readable.columns:
        X_test_readable[col] = le.inverse_transform(X_test_readable[col])

# Build explainer
explainer = RegressionExplainer(
    model,
    X_test_scaled,
    y_test_unscaled,
    X_test_orig=X_test_readable
)

# Launch dashboard
if __name__ == "__main__":
    ExplainerDashboard(explainer).run()
