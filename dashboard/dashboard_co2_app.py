import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# --- Données mises à jour selon vos métriques ---
data = {
    "Modèle": ["XGBoost", "Random Forest", "Decision Tree", "MLP (Neural)"],
    "MSE Entraînement": [1.953476, 41.985219, 123.090402, 51.324280],
    "MSE Test": [14.484346, 59.927690, 150.067544, 56.334942],
    "MSE Validation": [6.472137, 61.523784, 113.233513, 38.346434],
    
    "RMSE Entraînement": [1.397668, 6.479600, 11.094611, 7.164097],
    "RMSE Test": [3.805831, 7.741298, 12.250206, 7.505661],
    "RMSE Validation": [2.544039, 7.843710, 10.641124, 6.192450],
    
    "MAE Entraînement": [1.014017, 3.941177, 6.697855, 3.796159],
    "MAE Test": [1.649261, 4.625895, 6.915347, 3.924296],
    "MAE Validation": [1.495198, 4.684207, 6.751206, 3.616577],
    
    "R² Entraînement": [0.999505, 0.989360, 0.968807, 0.986994],
    "R² Test": [0.996189, 0.984232, 0.960514, 0.985177],
    "R² Validation": [0.998314, 0.983975, 0.970506, 0.990012]
}

df = pd.DataFrame(data)

# --- Bar plot pour visualiser RMSE Test ---
fig = px.bar(df, x="Modèle", y="RMSE Test", title="Comparaison RMSE Test", text="RMSE Test")
fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')

# --- Layout Dash ---
app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1("Dashboard des Modèles CO₂"),
    
    html.H2("Comparaison RMSE Test"),
    dcc.Graph(figure=fig),
    
    html.H2("Tableau des performances complètes"),
    html.Table([
        # Header
        html.Tr([html.Th(col) for col in df.columns])
    ] + [
        # Rows
        html.Tr([html.Td(f"{df.iloc[i][col]:.3f}" if isinstance(df.iloc[i][col], float) else df.iloc[i][col]) 
                 for col in df.columns])
        for i in range(len(df))
    ], style={'border': '1px solid black', 'border-collapse': 'collapse', 'width': '100%'})
])

if __name__ == '__main__':
    app.run(debug=True)
