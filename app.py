import pandas as pd
import joblib
import os
from explainerdashboard import RegressionExplainer, ExplainerDashboard

# Load model and data (limit to 1000 rows for faster load)
model = joblib.load("best_xgb_model.pkl")
X_test_scaled = joblib.load("X_test_scaled.pkl")
y_test = joblib.load("y_test.pkl")

explainer = RegressionExplainer(
    model,
    X_test_scaled[:100],
    y_test[:100]
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print(f"Starting ExplainerDashboard on port {port}")
    ExplainerDashboard(
        explainer,
        shap_interaction=False,  # disables SHAP interaction values
        no_permutations=True,    # skips permutation importance
        simple=True              # loads only basic tabs
    ).run(port=port, host="0.0.0.0")
