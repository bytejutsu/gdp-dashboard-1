from pathlib import Path
import folium
from streamlit_folium import st_folium
import streamlit as st
import pandas as pd
from io import StringIO
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------------------------
# 1. Configuration de la Page Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title='SafeBuddy Dashboard',
    page_icon=':earth_americas:',  # Emoji ou URL
    layout='wide',  # Optionnel : pour une disposition large
)

# -----------------------------------------------------------------------------
# 2. Titre et Description de l'Application
# -----------------------------------------------------------------------------
st.markdown("""
# :earth_americas: SafeBuddy Dashboard
Un tableau de bord incroyable utilisant le ML qui affiche les risques potentiels de sécurité et de sûreté afin de les atténuer.
""")

st.title("Crime Incidents in the Governorate of Tunis (Mock Data)")
st.write("""
Cette application **Streamlit** démontre comment nous pourrions explorer et visualiser un jeu de données
de dossiers d'incidents criminels fictifs basés sur :
- **Âge et sexe de la personne**
- **Localisation (latitude, longitude, zone)**
- **Heure et jour de la semaine**
- **Incident ou non (binaire)**

Toutes les données sont **totalement fictives** et uniquement à des fins de démonstration.
""")

# -----------------------------------------------------------------------------
# 3. Chargement et Préparation des Données
# -----------------------------------------------------------------------------
DATA_FILENAME = Path(__file__).parent / 'data/crime_data.csv'
df = pd.read_csv(DATA_FILENAME)

# Renommer les colonnes si nécessaire pour correspondre aux attentes de Streamlit
df.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True)

# Optionnel : Traiter les valeurs manquantes et convertir les types si nécessaire
df.dropna(subset=['latitude', 'longitude'], inplace=True)
df['latitude'] = df['latitude'].astype(float)
df['longitude'] = df['longitude'].astype(float)

# Ajouter une colonne 'Hour' pour filtrer par heure de la journée
df['Hour'] = df['TimeOfDay'].str.split(':').apply(lambda x: int(x[0]))

# -----------------------------------------------------------------------------
# 4. Filtres dans la Sidebar
# -----------------------------------------------------------------------------
st.sidebar.header("Filter Data")

# 4.1 Sélectionner le(s) jour(s) de la semaine
all_days = sorted(df['DayOfWeek'].unique())
selected_days = st.sidebar.multiselect(
    "Select Day(s) of Week",
    options=all_days,
    default=all_days
)

# 4.2 Sélectionner la ou les zones
all_zones = sorted(df['Zone'].unique())
selected_zones = st.sidebar.multiselect(
    "Select Zone(s)",
    options=all_zones,
    default=all_zones
)

# 4.3 Filtrer par plage horaire (heure uniquement)
min_hour, max_hour = st.sidebar.slider(
    "Filter by Hour Range (24-hour format)",
    min_value=0,
    max_value=23,
    value=(0, 23)
)

# 4.4 Appliquer les filtres
filtered_df = df[
    (df['DayOfWeek'].isin(selected_days)) &
    (df['Zone'].isin(selected_zones)) &
    (df['Hour'] >= min_hour) &
    (df['Hour'] <= max_hour)
    ].copy()

# -----------------------------------------------------------------------------
# 5. Vue d'Ensemble des Données Filtrées
# -----------------------------------------------------------------------------
st.subheader("Filtered Data Overview")
st.write(f"**Number of records in filtered dataset: {filtered_df.shape[0]}**")
st.dataframe(filtered_df)

# -----------------------------------------------------------------------------
# 6. Visualisations
# -----------------------------------------------------------------------------
st.subheader("Visualizations")

# 6.1 Incidents par Zone (Graphique en barres)
st.markdown("### Incidents by Zone")
incident_counts = (
    filtered_df.groupby(['Zone', 'IncidentHappened'])
    .size()
    .reset_index(name='count')
)

chart_incidents_by_zone = alt.Chart(incident_counts).mark_bar().encode(
    x=alt.X('Zone:N', sort='-y', title='Zone'),
    y=alt.Y('count:Q', title='Count of Records'),
    color=alt.Color('IncidentHappened:N', title='Incident Happened?'),
    tooltip=['Zone', 'IncidentHappened', 'count']
).properties(
    width=600,
    height=400
)

st.altair_chart(chart_incidents_by_zone, use_container_width=True)

# 6.2 Incidents par Heure de la Journée (Graphique en lignes)
st.markdown("### Incidents by Hour of Day")
incidents_by_hour = (
    filtered_df.groupby(['Hour', 'IncidentHappened'])
    .size()
    .reset_index(name='count')
)

line_chart = alt.Chart(incidents_by_hour).mark_line().encode(
    x=alt.X('Hour:O', title='Hour of Day'),
    y=alt.Y('count:Q', title='Count of Records'),
    color='IncidentHappened:N',
    tooltip=['Hour', 'IncidentHappened', 'count']
).properties(
    width=600,
    height=400
)

st.altair_chart(line_chart, use_container_width=True)

# 6.3 Carte des Incidents (Latitude & Longitude)
st.markdown("### Map of Locations (Latitude & Longitude)")

# Convertir 'IncidentHappened' en chaîne pour un codage couleur optionnel
filtered_df['IncidentHappenedStr'] = filtered_df['IncidentHappened'].apply(
    lambda x: "Incident" if x == 1 else "No Incident"
)

# Vérifier si le DataFrame filtré contient des données avant d'afficher la carte
if not filtered_df.empty:
    st.map(filtered_df[['latitude', 'longitude']])
else:
    st.warning("Aucune donnée disponible pour les filtres sélectionnés.")

# -----------------------------------------------------------------------------
# 7. Modélisation et Prédiction ML
# -----------------------------------------------------------------------------
st.subheader("Machine Learning Model: Predict Incident Happened")

# 7.1 Préparation des Données pour le Modèle
# Sélectionner les features et la cible
features = ['PersonAge', 'PersonSex', 'TimeOfDay', 'DayOfWeek', 'latitude', 'longitude', 'Zone']
target = 'IncidentHappened'

# Encodage des variables catégorielles
le_sex = LabelEncoder()
le_day = LabelEncoder()
le_zone = LabelEncoder()

df_model = df.copy()
df_model['PersonSexEncoded'] = le_sex.fit_transform(df_model['PersonSex'])
df_model['DayOfWeekEncoded'] = le_day.fit_transform(df_model['DayOfWeek'])
df_model['ZoneEncoded'] = le_zone.fit_transform(df_model['Zone'])

# Convertir TimeOfDay en minutes depuis minuit pour une meilleure interprétation
df_model['TimeOfDayMinutes'] = df_model['TimeOfDay'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))

# Définir les features finales
X = df_model[
    ['PersonAge', 'PersonSexEncoded', 'DayOfWeekEncoded', 'TimeOfDayMinutes', 'latitude', 'longitude', 'ZoneEncoded']]
y = df_model['IncidentHappened']

# 7.2 Entraînement du Modèle
# Séparer les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser et entraîner le modèle de décision
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 7.3 Évaluation du Modèle
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

st.write(f"**Model Accuracy:** {accuracy:.2f}")

# Afficher le rapport de classification
report_df = pd.DataFrame(report).transpose()
st.write("**Classification Report:**")
st.dataframe(report_df)

# 7.4 Interface Utilisateur pour la Prédiction
st.markdown("### Predict New Incident")

# Création des inputs pour la prédiction
col1, col2, col3 = st.columns(3)

with col1:
    input_age = st.number_input("Person Age", min_value=0, max_value=120, value=30)

with col2:
    input_sex = st.selectbox("Person Sex", options=['M', 'F'])

with col3:
    input_day = st.selectbox("Day of Week", options=sorted(df['DayOfWeek'].unique()))

with col1:
    input_zone = st.selectbox("Zone", options=sorted(df['Zone'].unique()))

with col2:
    input_time = st.time_input("Time of Day", value=pd.to_datetime("12:00").time())

with col3:
    input_latitude = st.number_input("latitude", value=36.8)
    input_longitude = st.number_input("longitude", value=10.1)

# Bouton de prédiction
if st.button("Predict Incident"):
    # Préparer les données d'entrée
    input_sex_encoded = le_sex.transform([input_sex])[0]
    input_day_encoded = le_day.transform([input_day])[0]
    input_zone_encoded = le_zone.transform([input_zone])[0]
    input_time_minutes = input_time.hour * 60 + input_time.minute

    input_data = pd.DataFrame({
        'PersonAge': [input_age],
        'PersonSexEncoded': [input_sex_encoded],
        'DayOfWeekEncoded': [input_day_encoded],
        'TimeOfDayMinutes': [input_time_minutes],
        'latitude': [input_latitude],
        'longitude': [input_longitude],
        'ZoneEncoded': [input_zone_encoded]
    })

    # Prédire
    prediction = clf.predict(input_data)[0]
    prediction_proba = clf.predict_proba(input_data)[0]

    # Interprétation de la prédiction
    if prediction == 1:
        st.success("**Prediction:** An incident is likely to happen.")
    else:
        st.info("**Prediction:** No incident is likely to happen.")

    st.write(f"**Probability of No Incident:** {prediction_proba[0]:.2f}")
    st.write(f"**Probability of Incident:** {prediction_proba[1]:.2f}")

# -----------------------------------------------------------------------------
# 8. Exploration Supplémentaire
# -----------------------------------------------------------------------------
st.subheader("Incidents by Age and Sex")
sex_age_counts = (
    filtered_df.groupby(['PersonSex', 'PersonAge', 'IncidentHappened'])
    .size()
    .reset_index(name='count')
)

# Afficher dans un tableau croisé dynamique
pivot_sex_age = sex_age_counts.pivot_table(
    index=['PersonSex', 'PersonAge'],
    columns='IncidentHappened',
    values='count',
    aggfunc='sum',
    fill_value=0
).rename(columns={0: 'No Incident', 1: 'Incident'})

st.dataframe(pivot_sex_age)

st.write("""
**Note** : Avec un vrai jeu de données, vous pourriez construire des visualisations plus avancées
ou des modèles prédictifs en utilisant les fonctionnalités ci-dessus.
""")

# -----------------------------------------------------------------------------
# 9. (Optionnel) Carte Folium Supplémentaire
# -----------------------------------------------------------------------------
# Si vous souhaitez utiliser Folium pour des visualisations de cartes plus avancées,
# vous pouvez décommenter et utiliser le code suivant.

# st.markdown("### Folium Map of Incidents")
# if not filtered_df.empty:
#     # Centrer la carte sur la moyenne des latitudes et longitudes filtrées
#     mean_lat = filtered_df['latitude'].mean()
#     mean_lon = filtered_df['longitude'].mean()
#     m_incidents = folium.Map(location=[mean_lat, mean_lon], zoom_start=12)

#     # Ajouter des marqueurs pour chaque incident
#     for idx, row in filtered_df.iterrows():
#         folium.Marker(
#             location=[row['latitude'], row['longitude']],
#             popup=f"Incident: {row['IncidentHappenedStr']}",
#             tooltip=row['IncidentHappenedStr']
#         ).add_to(m_incidents)

#     # Rendre la carte Folium dans Streamlit
#     st_folium(m_incidents, width=700)
# else:
#     st.warning("Aucune donnée disponible pour les filtres sélectionnés.")