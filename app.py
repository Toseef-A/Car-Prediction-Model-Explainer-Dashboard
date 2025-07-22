from dash import Dash, html
import os

app = Dash(__name__)
app.layout = html.Div("Test App")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print(f"Running test app on port {port}")
    app.run_server(host="0.0.0.0", port=port)
