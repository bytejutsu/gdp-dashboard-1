from pathlib import Path
import folium
from streamlit_folium import st_folium
import streamlit as st
import pandas as pd
from io import StringIO
import altair as alt

# Configurer la page Streamlit
st.set_page_config(
    page_title='SafeBuddy Dashboard',
    page_icon=':earth_americas:',  # Emoji ou URL
)

# -----------------------------------------------------------------------------
# Dessiner la page principale

# Titre de la page
'''
# :earth_americas: SafeBuddy Dashboard
Un tableau de bord incroyable utilisant le ML qui affiche les risques potentiels de sécurité et de sûreté afin de les atténuer.
'''

# Centrer sur Liberty Bell et ajouter un marqueur
#m = folium.Map(location=[36.847175, 10.199055], zoom_start=13)
#folium.Marker(
#    [36.847175, 10.199055], popup="UIT", tooltip="Current Location"
#).add_to(m)

# Rendre la carte Folium dans Streamlit
# st_data = st_folium(m, width=725)
# Charger les données CSV
DATA_FILENAME = Path(__file__).parent / 'data/crime_data.csv'
df = pd.read_csv(DATA_FILENAME)

# Renommer les colonnes si nécessaire
df.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True)

# Optionnel : Traiter les valeurs manquantes et convertir les types si nécessaire
df.dropna(subset=['latitude', 'longitude'], inplace=True)
df['latitude'] = df['latitude'].astype(float)
df['longitude'] = df['longitude'].astype(float)

# --------------------------------------------------
# 4. Filtres dans la Sidebar
# --------------------------------------------------
st.sidebar.header("Filter Data")

# 4.1 Sélectionner le(s) jour(s) de la semaine
all_days = sorted(df['DayOfWeek'].unique())
selected_days = st.sidebar.multiselect("Select Day(s) of Week", options=all_days, default=all_days)

# 4.2 Sélectionner la ou les zones
all_zones = sorted(df['Zone'].unique())
selected_zones = st.sidebar.multiselect("Select Zone(s)", options=all_zones, default=all_zones)

# 4.3 Filtrer par plage horaire
# Convertir les chaînes de temps en entiers (heure uniquement)
df['Hour'] = df['TimeOfDay'].str.split(':').apply(lambda x: int(x[0]))
min_hour, max_hour = st.sidebar.slider("Filter by Hour Range (24-hour format)",
                                       min_value=0, max_value=23,
                                       value=(0, 23))

# 4.4 Appliquer les filtres
filtered_df = df[
    (df['DayOfWeek'].isin(selected_days)) &
    (df['Zone'].isin(selected_zones)) &
    (df['Hour'] >= min_hour) &
    (df['Hour'] <= max_hour)
].copy()

# 6.3 Carte des Incidents
st.markdown("### Map of Locations (Latitude & Longitude)")

# Convert IncidentHappened en chaîne pour le codage couleur (optionnel)
filtered_df['IncidentHappenedStr'] = filtered_df['IncidentHappened'].apply(
    lambda x: "Incident" if x == 1 else "No Incident"
)

# Vérifier si le DataFrame filtré contient des données avant d'afficher la carte
if not filtered_df.empty:
    st.map(filtered_df[['latitude', 'longitude']])
else:
    st.warning("Aucune donnée disponible pour les filtres sélectionnés.")

# -----------------------------------------------------------------------------

# --------------------------------------------------
# 3. Titre et Description de l'App
# --------------------------------------------------
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



# --------------------------------------------------
# 5. Vue d'ensemble des données
# --------------------------------------------------
st.subheader("Filtered Data Overview")
st.write(f"**Number of records in filtered dataset: {filtered_df.shape[0]}**")
st.dataframe(filtered_df)

# --------------------------------------------------
# 6. Visualisations
# --------------------------------------------------
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
).properties(width=600, height=400)

st.altair_chart(chart_incidents_by_zone, use_container_width=True)

# 6.2 Incident vs. Non-Incident par Heure (Graphique en lignes)
st.markdown("### Incidents by Hour of Day")
incidents_by_hour = (
    filtered_df.groupby(['Hour', 'IncidentHappened'])
    .size()
    .reset_index(name='count')
)

# Créer un graphique en lignes superposées pour les incidents et non-incidents
line_chart = alt.Chart(incidents_by_hour).mark_line().encode(
    x=alt.X('Hour:O', title='Hour of Day'),
    y=alt.Y('count:Q', title='Count of Records'),
    color='IncidentHappened:N',
    tooltip=['Hour', 'IncidentHappened', 'count']
).properties(width=600, height=400)

st.altair_chart(line_chart, use_container_width=True)

# --------------------------------------------------
# 7. Exploration Supplémentaire
# --------------------------------------------------
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