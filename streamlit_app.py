from pathlib import Path
import folium
from streamlit_folium import st_folium
import streamlit as st
import pandas as pd
from io import StringIO
import altair as alt

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='SafeBuddy dashboard',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :earth_americas: SafeBuddy Dashboard 
An Amazing Dashboard using ML that displays the potential security and safety risks to try to mitigate them.
'''


# center on Liberty Bell, add marker
m = folium.Map(location=[36.847175, 10.199055], zoom_start=13)
folium.Marker(
    [36.847175, 10.199055], popup="UIT", tooltip="Current Location"
).add_to(m)

# call to render Folium map in Streamlit
st_data = st_folium(m, width=725)

# -----------------------------------------------------------------------------

# Instead of a CSV on disk, you could read from an HTTP endpoint here too.
DATA_FILENAME = Path(__file__).parent/'data/crime_data.csv'
df = pd.read_csv(DATA_FILENAME)

# --------------------------------------------------
# 3. App Title and Description
# --------------------------------------------------
st.title("Crime Incidents in the Governorate of Tunis (Mock Data)")
st.write("""
This **Streamlit** app demonstrates how we might explore and visualize a dataset
of mock criminal-incident records based on:
- **Person's age and sex**
- **Location (latitude, longitude, zone)**
- **Time and day of week**
- **Incident or not (binary)**

All data is **completely fictitious** and for demonstration purposes only.
""")

# --------------------------------------------------
# 4. Sidebar Filters
# --------------------------------------------------
st.sidebar.header("Filter Data")

# 4.1 Select which Day(s) of the Week to view
all_days = sorted(df['DayOfWeek'].unique())
selected_days = st.sidebar.multiselect("Select Day(s) of Week", options=all_days, default=all_days)

# 4.2 Select which Zones to view
all_zones = sorted(df['Zone'].unique())
selected_zones = st.sidebar.multiselect("Select Zone(s)", options=all_zones, default=all_zones)

# 4.3 Filter by TimeOfDay Range
#    Convert time strings to int for simpler filtering (just hour, ignoring minutes)
df['Hour'] = df['TimeOfDay'].str.split(':').apply(lambda x: int(x[0]))
min_hour, max_hour = st.sidebar.slider("Filter by Hour Range (24-hour format)",
                                       min_value=0, max_value=23,
                                       value=(0, 23))

# 4.4 Apply filters
filtered_df = df[
    (df['DayOfWeek'].isin(selected_days)) &
    (df['Zone'].isin(selected_zones)) &
    (df['Hour'] >= min_hour) &
    (df['Hour'] <= max_hour)
].copy()

# --------------------------------------------------
# 5. Data Overview
# --------------------------------------------------
st.subheader("Filtered Data Overview")
st.write(f"**Number of records in filtered dataset: {filtered_df.shape[0]}**")
st.dataframe(filtered_df)

# --------------------------------------------------
# 6. Visualizations
# --------------------------------------------------
st.subheader("Visualizations")

# 6.1 Incidents by Zone (Bar Chart)
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

# 6.2 Incident vs. Non-Incident by Hour (Line/Area Chart)
st.markdown("### Incidents by Hour of Day")
incidents_by_hour = (
    filtered_df.groupby(['Hour', 'IncidentHappened'])
    .size()
    .reset_index(name='count')
)

# Create a layered chart to show line for 0 vs 1 incidents
line_chart = alt.Chart(incidents_by_hour).mark_line().encode(
    x=alt.X('Hour:O', title='Hour of Day'),
    y=alt.Y('count:Q', title='Count of Records'),
    color='IncidentHappened:N',
    tooltip=['Hour', 'IncidentHappened', 'count']
).properties(width=600, height=400)

st.altair_chart(line_chart, use_container_width=True)

# 6.3 Map Plot of Incidents
st.markdown("### Map of Locations (Latitude & Longitude)")

# Convert IncidentHappened to string for color coding in map
filtered_df['IncidentHappenedStr'] = filtered_df['IncidentHappened'].apply(
    lambda x: "Incident" if x == 1 else "No Incident"
)

st.map(filtered_df[['Latitude', 'Longitude']])

# --------------------------------------------------
# 7. Additional Exploration
# --------------------------------------------------
st.subheader("Incidents by Age and Sex")
sex_age_counts = (
    filtered_df.groupby(['PersonSex', 'PersonAge', 'IncidentHappened'])
    .size()
    .reset_index(name='count')
)

# We'll display this in a pivot table style
pivot_sex_age = sex_age_counts.pivot_table(
    index=['PersonSex', 'PersonAge'],
    columns='IncidentHappened',
    values='count',
    aggfunc='sum',
    fill_value=0
).rename(columns={0: 'No Incident', 1: 'Incident'})

st.dataframe(pivot_sex_age)

st.write("""
**Note**: With a real dataset, you could build more advanced visualizations
or predictive models using the above features.
""")