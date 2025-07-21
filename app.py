%%writefile app.py
import pandas as pd
import joblib
from xgboost import XGBRegressor
from explainerdashboard import RegressionExplainer, ExplainerDashboard

# Load the trained model
model = joblib.load("xgb_model.pkl")

# Load the scaled test data and labels
X_test_scaled = joblib.load("X_test_scaled.pkl")
y_test = joblib.load("y_test.pkl")

# Create the explainer
explainer = RegressionExplainer(model, X_test_scaled, y_test)

# Launch the dashboard
if __name__ == "__main__":
    ExplainerDashboard(explainer).run()
