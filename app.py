import pandas as pd
import joblib
from explainerdashboard import RegressionExplainer, ExplainerDashboard

# Load model and data
model = joblib.load("best_xgb_model.pkl")
X_test_scaled = joblib.load("X_test_scaled.pkl")
y_test = joblib.load("y_test.pkl")

# Create the explainer using the scaled test data
explainer = RegressionExplainer(model, X_test_scaled, y_test)

# Launch the dashboard
if __name__ == "__main__":
    ExplainerDashboard(explainer).run()
