import pandas as pd
import joblib
from explainerdashboard import RegressionExplainer, ExplainerDashboard
import os

# Load model and data
model = joblib.load("best_xgb_model.pkl")
X_test_scaled = joblib.load("X_test_scaled.pkl")
y_test = joblib.load("y_test.pkl")

# Create the explainer
explainer = RegressionExplainer(model, X_test_scaled, y_test)

# Run the dashboard on the correct port for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    ExplainerDashboard(explainer).run(port=port)
