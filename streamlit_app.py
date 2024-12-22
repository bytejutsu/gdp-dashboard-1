from pathlib import Path
import folium
from streamlit_folium import st_folium
import streamlit as st
import pandas as pd
from io import StringIO
import altair as alt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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
@st.cache_data
def load_data(data_path):
    df = pd.read_csv(data_path)
    df.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True)
    df.dropna(subset=['latitude', 'longitude'], inplace=True)
    df['latitude'] = df['latitude'].astype(float)
    df['longitude'] = df['longitude'].astype(float)
    df['Hour'] = df['TimeOfDay'].str.split(':').apply(lambda x: int(x[0]))
    df = pd.get_dummies(df, columns=['DayOfWeek', 'Zone', 'PersonSex'], drop_first=True)
    return df


DATA_FILENAME = Path(__file__).parent / 'data/crime_data.csv'
df = load_data(DATA_FILENAME)

# Définir la variable cible et les caractéristiques
X = df.drop(['IncidentHappened', 'TimeOfDay', 'PersonAge'], axis=1)
y = df['IncidentHappened']

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------------------------------------
# 4. Filtres dans la Sidebar
# -----------------------------------------------------------------------------
st.sidebar.header("Filter Data")

# 4.1 Sélectionner le(s) jour(s) de la semaine
day_options = [col.replace('DayOfWeek_', '') for col in X.columns if 'DayOfWeek_' in col]
selected_days = st.sidebar.multiselect(
    "Select Day(s) of Week",
    options=day_options,
    default=day_options
)

# 4.2 Sélectionner la ou les zones
zone_options = [col.replace('Zone_', '') for col in X.columns if 'Zone_' in col]
selected_zones = st.sidebar.multiselect(
    "Select Zone(s)",
    options=zone_options,
    default=zone_options
)

# 4.3 Filtrer par plage horaire (heure uniquement)
min_hour, max_hour = st.sidebar.slider(
    "Filter by Hour Range (24-hour format)",
    min_value=0,
    max_value=23,
    value=(0, 23)
)

# 4.4 Appliquer les filtres
# Construire les sélections de colonnes pour les jours et zones
day_columns = [f'DayOfWeek_{day}' for day in selected_days]
zone_columns = [f'Zone_{zone}' for zone in selected_zones]

filtered_df = df[
    (df['Hour'] >= min_hour) &
    (df['Hour'] <= max_hour)
    ].copy()

# Sélectionner les colonnes correspondant aux jours sélectionnés
if day_columns:
    filtered_df = filtered_df[filtered_df[day_columns].sum(axis=1) > 0]

# Sélectionner les colonnes correspondant aux zones sélectionnées
if zone_columns:
    filtered_df = filtered_df[filtered_df[zone_columns].sum(axis=1) > 0]

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
# Remplacez 'Zone_<Zone1>' par vos zones réelles
incident_counts = (
    filtered_df.groupby(['Zone_<Zone1>', 'IncidentHappened'])
    .size()
    .reset_index(name='count')
)

chart_incidents_by_zone = alt.Chart(incident_counts).mark_bar().encode(
    x=alt.X('Zone_<Zone1>:N', sort='-y', title='Zone'),  # Ajustez selon vos zones réelles
    y=alt.Y('count:Q', title='Count of Records'),
    color=alt.Color('IncidentHappened:N', title='Incident Happened?'),
    tooltip=['Zone_<Zone1>', 'IncidentHappened', 'count']  # Ajustez selon vos zones réelles
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
# 7. Exploration Supplémentaire
# -----------------------------------------------------------------------------
st.subheader("Incidents by Age and Sex")
sex_age_counts = (
    filtered_df.groupby(['PersonSex_Male', 'PersonAge', 'IncidentHappened'])
    .size()
    .reset_index(name='count')
)

# Afficher dans un tableau croisé dynamique
pivot_sex_age = sex_age_counts.pivot_table(
    index=['PersonSex_Male', 'PersonAge'],
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
# 8. Prédiction des Incidents
# -----------------------------------------------------------------------------
st.sidebar.header("Predict New Incident")


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


model = load_model(Path(__file__).parent / 'model/random_forest_model.joblib')

with st.sidebar.form(key='prediction_form'):
    st.write("### Enter Details to Predict Incident")

    # Entrées utilisateur
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Sex", options=['Male', 'Female'])
    day_of_week = st.selectbox("Day of Week", options=day_options)
    zone = st.selectbox("Zone", options=zone_options)
    hour = st.slider("Hour of Day", min_value=0, max_value=23, value=12)
    latitude = st.number_input("Latitude", value=36.847175)
    longitude = st.number_input("Longitude", value=10.199055)

    # Soumettre le formulaire
    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    # Préparer les données d'entrée pour le modèle
    input_data = {
        'Hour': hour,
        'latitude': latitude,
        'longitude': longitude,
        'PersonAge': age,
        'PersonSex_Male': 1 if sex == 'Male' else 0,
        'DayOfWeek_Monday': 1 if day_of_week == 'Monday' else 0,
        'DayOfWeek_Saturday': 1 if day_of_week == 'Saturday' else 0,
        'DayOfWeek_Sunday': 1 if day_of_week == 'Sunday' else 0,
        'DayOfWeek_Thursday': 1 if day_of_week == 'Thursday' else 0,
        'DayOfWeek_Tuesday': 1 if day_of_week == 'Tuesday' else 0,
        'DayOfWeek_Wednesday': 1 if day_of_week == 'Wednesday' else 0,
        'Zone_<Zone1>': 1 if zone == '<Zone1>' else 0,  # Remplacez <Zone1> par vos zones réelles
        'Zone_<Zone2>': 1 if zone == '<Zone2>' else 0,
        # Ajoutez d'autres zones selon votre dataset
    }

    # Convertir en DataFrame
    input_df = pd.DataFrame([input_data])

    # Assurez-vous que toutes les colonnes du modèle sont présentes
    model_features = X.columns
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0

    # Réorganiser les colonnes
    input_df = input_df[model_features]

    # Faire la prédiction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    # Afficher le résultat
    if prediction == 1:
        st.success(f"**Prediction:** Incident is likely to occur. (Probability: {prediction_proba:.2f})")
    else:
        st.info(f"**Prediction:** No incident is likely to occur. (Probability: {1 - prediction_proba:.2f})")