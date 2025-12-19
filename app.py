import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# --- CONFIGURATION ---
DATASET_PATH = 'dataset_final.csv'

# Liste des modèles
MODEL_FILES = {
    'Random Forest': 'pickle/random_forest.pkl',
    'XGBoost': 'pickle/xgboost.pkl',
    'MLP (Neural Net)': 'pickle/mlp.pkl'
}

# Mapping Carburants (CORRIGÉ selon votre dataset)
FUEL_MAPPING = {
    'Autre': 'Autre',
    'Diesel': 'Diesel',
    'Electrique': 'Électrique',
    'Essence': 'Essence',
    'Essence-Electricité': 'Essence-Électricité (Hybride)',
    'Ethanol': 'Éthanol (E85)',
    'GNV': 'Gaz Naturel (GNV)',
    'GPL': 'GPL',
    'Gazole-Electricité': 'Gazole-Électricité (Hybride)'
}

# Mapping Transmissions simplifié pour l'interface
TRANSMISSION_MAPPING = {
    'A': 'Automatique',
    'M': 'Manuelle',
    'AM': 'Automatique Manuelle',
    'AS': 'Automatique Séquentielle',
    'AV': 'Automatique Variable',
    'D': 'Double Embrayage',
    'S': 'Séquentielle',
    'V': 'Variable',
    'N': 'Neutre'
}

# Variables globales
loaded_models = {}
feature_columns_csv = []
marques = []
carburants_options = []
transmissions_options = []

def load_resources():
    global loaded_models, feature_columns_csv
    global marques, carburants_options, transmissions_options, model_columns
    
    print("--- Chargement des ressources ---")

    # 1. Modèles
    for name, filename in MODEL_FILES.items():
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    loaded_models[name] = pickle.load(f)
                print(f"✓ {name} chargé.")
            except Exception as e:
                print(f"✗ Erreur {name}: {e}")



    # 4. Dataset (Pour les listes déroulantes et RF/XGB)
    if os.path.exists(DATASET_PATH):
        try:
            df = pd.read_csv(DATASET_PATH)
            target_col = 'remainder__Emissions CO2 (g/km)'
            feature_columns_csv = [col for col in df.columns if col != target_col and col != 'Unnamed: 0']
            
            # Extraction Marques
            marques_cols = [c for c in feature_columns_csv if 'ohe__Marque_' in c]
            marques = sorted([c.replace('ohe__Marque_', '') for c in marques_cols])
            
            # Extraction Carburants (SANS les "ohe__Type de Carburant_")
            fuel_cols = [c for c in feature_columns_csv if 'ohe__Type de Carburant_' in c]
            raw_fuels = sorted([c.replace('ohe__Type de Carburant_', '') for c in fuel_cols])
            carburants_options = [(code, FUEL_MAPPING.get(code, code)) for code in raw_fuels]
            
            # Extraction Transmissions (codes uniques)
            trans_cols = [c for c in feature_columns_csv if 'ohe__Transmission_' in c]
            raw_trans = sorted(set([c.replace('ohe__Transmission_', '') for c in trans_cols]))
            transmissions_options = raw_trans
            
            print(f"✓ {len(marques)} marques, {len(carburants_options)} carburants, {len(transmissions_options)} transmissions chargés.")
            
        except Exception as e:
            print(f"✗ Erreur CSV: {e}")

load_resources()

@app.route('/')
def home():
    return render_template('index.html', 
                         marques=marques, 
                         carburants=carburants_options,
                         transmissions=transmissions_options)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form
        input_data = {}
        
        # --- 1. MAPPING DES VARIABLES (MISE À JOUR) ---
        mapping_numerique = {
            'remainder__Consommation Urbaine (L/100 km)': 'fuel_city',
            'remainder__Consommation Autoroute (L/100 km)': 'fuel_highway',
            'remainder__Puissance Moteur': 'engine_power',
            'remainder__Puissance Fiscale (CV)': 'engine_fiscal',
            'remainder__Cylindres': 'cylinders',
            'remainder__Cylindrée (L)': 'cylindree',
            'remainder__Masse Min (kg)': 'mass_min',
            'remainder__Masse Max (kg)': 'mass_max',
            'remainder__Age_Vehicule': 'age'
        }
        
        for col_model, input_name in mapping_numerique.items():
            val = form_data.get(input_name)
            input_data[col_model] = float(val) if val else 0.0

        # Catégories - Marque
        if form_data.get('marque'):
            input_data[f"ohe__Marque_{form_data.get('marque')}"] = 1.0
            
        # Catégories - Carburant
        carb_code = form_data.get('carburant')
        if carb_code:
            input_data[f"ohe__Type de Carburant_{carb_code}"] = 1.0
            
        # Catégories - Transmission
        trans_code = form_data.get('transmission')
        if trans_code:
            input_data[f"ohe__Transmission_{trans_code}"] = 1.0
            
        # Catégories - Hybride
        is_hybrid = form_data.get('is_hybrid', 'non')
        if is_hybrid == 'oui':
            input_data['ohe__Hybride_oui'] = 1.0
            input_data['ohe__Hybride_non'] = 0.0
        else:
            input_data['ohe__Hybride_oui'] = 0.0
            input_data['ohe__Hybride_non'] = 1.0
            
        # Logique Électrique -> Conso 0
        if carb_code == 'Electrique':
            input_data['remainder__Consommation Urbaine (L/100 km)'] = 0.0
            input_data['remainder__Consommation Autoroute (L/100 km)'] = 0.0

        # --- 2. PRÉDICTION MULTI-MODÈLES ---
        predictions_details = {}
        somme_predictions = 0
        nb_valides = 0
        
        for model_name, model_obj in loaded_models.items():
            try:
                features = [input_data.get(col, 0.0) for col in feature_columns_csv]
                pred = model_obj.predict(np.array([features]))[0]

                predictions_details[model_name] = round(pred, 2)
                somme_predictions += pred
                nb_valides += 1
            except Exception as e:
                print(f"Erreur prediction {model_name}: {e}")
                predictions_details[model_name] = "Err"
            

        final_prediction = round(somme_predictions / nb_valides, 2) if nb_valides > 0 else 0
        
        # Affichage du contexte
        carb_label = FUEL_MAPPING.get(carb_code, carb_code)
        trans_label = TRANSMISSION_MAPPING.get(trans_code[:1] if trans_code else '', trans_code)
        context_str = f"{form_data.get('marque')} • {carb_label} • {trans_label} • {form_data.get('engine_power')} ch"

        return render_template('index.html', 
                             marques=marques, 
                             carburants=carburants_options,
                             transmissions=transmissions_options,
                             final_prediction=final_prediction,
                             individual_predictions=predictions_details,
                             context=context_str,
                             scroll_to_result=True)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('index.html', 
                             marques=marques, 
                             carburants=carburants_options,
                             transmissions=transmissions_options,
                             error_msg="Erreur technique")

if __name__ == "__main__":
    app.run(debug=True)