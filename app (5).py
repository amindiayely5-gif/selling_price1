import streamlit as st
import joblib as jb
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Prédiction du prix d'une voiture",
    page_icon="🚗",
    layout="centered"
)
st.title("🚗 Prédiction du Prix de Vente d'un Véhicule")
st.write("Cette application utilise un modèle XGBoost pour prédire le prix de vente d'une voiture.")

# 2. CHARGEMENT DES FICHIERS
@st.cache_resource
def load_models():
    model = jb.load("xgb_model.joblib")
    scaler = jb.load("scaler.joblib")
    encoders = jb.load("encoders.joblib")
    uniques = jb.load("uniques.joblib")
    return model, scaler, encoders, uniques

try:
    xgb_model, scaler, encoders, uniques = load_models()
except Exception as e:
    st.error(f"Erreur lors du chargement des modèles : {e}. Assurez-vous que 'xgb_model.joblib', 'encoders.joblib', 'scaler.joblib' et 'uniques.joblib' sont dans le même dossier.")
    st.stop()

# 3. Formulaire d'entrée unique
st.subheader("Estimer un véhicule unique")

Kms_Driven = st.number_input("Kilométrage (Kms Driven)", min_value=0, value=50000)
Present_Price = st.number_input("Prix d'origine (en k$)", min_value=0.0, value=10.0)
fuel_type_val = st.selectbox("Type de Carburant", uniques[0])
seller_type_val = st.selectbox("Type de Vendeur", uniques[1])
transmission_val = st.selectbox("Boîte de vitesse", uniques[2])
age_val = st.number_input("Âge du véhicule (Années)", min_value=0, value=5)

if st.button("Prédire le prix"):
    try:
        # Encodage
        fuel_type_encoded = encoders[0].transform([fuel_type_val])[0]
        seller_type_encoded = encoders[1].transform([seller_type_val])[0]
        transmission_encoded = encoders[2].transform([transmission_val])[0]
        
        # Préparation du vecteur
        x_new = np.array([Kms_Driven, Present_Price, fuel_type_encoded, seller_type_encoded, transmission_encoded, age_val]).reshape(1, -1)
        
        # Prédiction
        y_pred = xgb_model.predict(x_new)[0]
        
        st.success(f"Le prix de vente estimé est de : **{round(float(y_pred), 2)} k$**")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
