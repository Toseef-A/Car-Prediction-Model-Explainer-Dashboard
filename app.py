import pandas as pd
import joblib
import os
from explainerdashboard import RegressionExplainer, ExplainerDashboard

# Load model and data
model = joblib.load("best_xgb_model.pkl")
X_test_scaled = joblib.load("X_test_scaled.pkl")
y_test = joblib.load("y_test.pkl")

# Build explainer
explainer = RegressionExplainer(
    model,
    X_test_scaled[:1000],
    y_test[:1000]
)

# Get port from environment (Render requirement)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8100))
    print(f"Running dashboard on port {port}")
    ExplainerDashboard(explainer, shap_interaction=False).run(port=port, host="0.0.0.0")
