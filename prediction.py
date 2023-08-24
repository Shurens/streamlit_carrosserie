import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import joblib

data = pd.read_csv('data_model.csv')

encoder = LabelEncoder()
data['Carrosserie'] = encoder.fit_transform(data['Carrosserie'])

New_Data = data[['Carrosserie', 'masse_ordma_min', 'masse_ordma_max', 'co2']]

X = data[['Carrosserie', 'masse_ordma_min', 'masse_ordma_max']]
y = data['co2']


model = joblib.load('meilleur_modele.pkl')


voiture = encoder.inverse_transform(X["Carrosserie"])

#Titre
st.title("Prédiction de CO2")

#Sélection du type de carrosserie 
carrosserie = st.selectbox("Sélectionnez le type de carrosserie:", np.unique(voiture))

#Saisie des masses
masse_min = st.number_input("Entrez la masse min:")
masse_max = st.number_input("Entrez la masse max:")

#Bouton pour lancer la prédiction
if st.button("Prédire CO2"):
    carrosserie_encode = encoder.transform([carrosserie])[0]
    
    #Préparation des données pour la prédiction
    data = np.array([carrosserie_encode, masse_min, masse_max]).reshape(1, -1)

    prediction = model.predict(data)
    
    #Résultat
    st.success(f"La prédiction de CO2 est : {prediction[0]:.2f} g/km")