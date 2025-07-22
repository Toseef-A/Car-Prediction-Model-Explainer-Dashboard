import pandas as pd
import joblib
import os
from explainerdashboard import RegressionExplainer, ExplainerDashboard

# Load model and data
model = joblib.load("best_xgb_model.pkl")
X_test_scaled = joblib.load("X_test_scaled.pkl")
y_test = joblib.load("y_test.pkl")

# Create explainer (limit to 1000 rows, disable SHAP interactions to speed up)
explainer = RegressionExplainer(
    model,
    X_test_scaled[:1000],
    y_test[:1000],
    shap_interaction=False
)

# Bind to dynamic port and 0.0.0.0 for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8100))
    print(f"Starting dashboard on port {port}...")
    ExplainerDashboard(explainer).run(port=port, host="0.0.0.0")
