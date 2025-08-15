import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page Configurations
st.set_page_config(page_title="Football Data Analysis", layout="wide")

# Define a function to load the data
@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_csv('NAC_Cleaned.csv')
    return data

# Define the categorize_position function
def categorize_position(position):
    if position in ['Forward', 'Striker']:
        return 'Attackers'
    elif position in ['Midfielder']:
        return 'Midfielders'
    else:
        return 'Defenders'

# Load the data
FootBall_Data = load_data()
FootBall_Data['Position_Category'] = FootBall_Data['Position'].apply(categorize_position)

# Algorithm 1: K-Means Clustering for Midfielders
st.header("Midfielders Passing Analysis")
# Select relevant passing metrics
passing_metrics = FootBall_Data[['Average pass length, m', 'Forward passes per 90', 'Accurate passes, %', 'Passes per 90']]

try:
    # Fit k-means model
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    FootBall_Data['Cluster'] = kmeans.fit_predict(passing_metrics)
    
    # Visualize the clusters (medium-sized and static graph)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Forward passes per 90', y='Average pass length, m', hue='Cluster', data=FootBall_Data, palette='viridis', s=100, ax=ax)
    plt.title('K-Means Clustering of Midfielders based on Passing Metrics')
    plt.xlabel('Forward passes per 90')
    plt.ylabel('Average pass length, m')
    st.pyplot(fig)
    
except Exception as e:
    st.error(f"An error occurred: {str(e)}")

# Algorithm 2: Defensive Players Analysis
st.header("Defensive Players Clustering")
# Filter for defensive players
defensive_players_df = FootBall_Data[(FootBall_Data['Position_Category'] == 'Defenders') & (FootBall_Data['Position'] != 'GK')]
# Selected defensive features
defensive_features = [
    "Successful defensive actions per 90",
    "Defensive duels per 90",
    "Defensive duels won, %",
    "Aerial duels per 90",
    "Aerial duels won, %",
    "Sliding tackles per 90",
    "Interceptions per 90",
    "Shots blocked per 90",
]
# Standardize the data
scaler = StandardScaler()
defensive_df_scaled = scaler.fit_transform(defensive_players_df[defensive_features])

try:
    # Apply K-Means clustering
    kmeans_defense = KMeans(n_clusters=3, n_init=10, random_state=42)
    defensive_players_df['Cluster'] = kmeans_defense.fit_predict(defensive_df_scaled)
    
    # Visualize the clusters (medium-sized and static graph)
    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster in range(3):
        cluster_data = defensive_players_df[defensive_players_df['Cluster'] == cluster]
        ax.scatter(cluster_data['Successful defensive actions per 90'], cluster_data['Defensive duels won, %'], label=f'Cluster {cluster}')
    plt.title('Defensive Clustering (Excluding Goalkeepers)')
    plt.xlabel('Successful defensive actions per 90')
    plt.ylabel('Defensive duels won %')
    plt.legend()
    st.pyplot(fig)
    
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
