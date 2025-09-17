import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Streamlit setup
st.set_page_config(page_title="Mall Customers Segmentation Demo", layout="wide")
sns.set(style="whitegrid")

# Load dataset
df = pd.read_csv("data/Mall_Customers.csv")
df['Genre_encoded'] = df['Genre'].map({'Male':0, 'Female':1})

# Features and scaling
features = ['Genre_encoded', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Load saved models
kmeans = joblib.load("models/kmeans_model.pkl")
dbscan = joblib.load("models/dbscan_model.pkl")

# Add cluster labels to DataFrame
df['KMeans_Cluster'] = kmeans.predict(X_scaled)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

# Sidebar: New customer input
st.sidebar.header("Add New Customer")
genre_input = st.sidebar.selectbox("Genre", ['Male', 'Female'])
age_input = st.sidebar.slider("Age", int(df['Age'].min()), int(df['Age'].max()), 30)
income_input = st.sidebar.slider("Annual Income (k$)", int(df['Annual Income (k$)'].min()), int(df['Annual Income (k$)'].max()), 60)
spending_input = st.sidebar.slider("Spending Score (1-100)", int(df['Spending Score (1-100)'].min()), int(df['Spending Score (1-100)'].max()), 50)

# Transform input
new_customer = np.array([[0 if genre_input=='Male' else 1, age_input, income_input, spending_input]])
new_customer_scaled = scaler.transform(new_customer)
new_customer_pca = pca.transform(new_customer_scaled)

# Predict clusters
kmeans_pred = kmeans.predict(new_customer_scaled)[0]
dbscan_pred = dbscan.fit_predict(new_customer_scaled)[0]

st.sidebar.markdown("### Cluster Prediction")
st.sidebar.write(f"KMeans Cluster: {kmeans_pred}")
st.sidebar.write(f"DBSCAN Cluster: {dbscan_pred}")

# Page title
st.title("Mall Customers Segmentation")

# Project description
st.markdown("""
### Project Description
This project segments mall customers into different groups based on their **demographics and spending behavior**. 
Using **KMeans** and **DBSCAN** clustering, we identify natural customer groups, which can help businesses with:

- **Targeted marketing campaigns**  
- **Personalized recommendations**  
- **Customer behavior analysis and business insights**

The project uses **PCA for 2D visualization** of clusters and allows you to **add a new customer** via the sidebar to predict their cluster in real-time.
""")

# Dataset preview
st.header("Dataset Preview")
st.dataframe(df)

# EDA Plots
st.header("Exploratory Data Analysis")
col1, col2, col3 = st.columns(3)

with col1:
    fig, ax = plt.subplots(figsize=(4,3))
    sns.countplot(data=df, x='Genre', palette='Set2', ax=ax)
    ax.set_title("Genre Count", fontsize=10)
    st.pyplot(fig, clear_figure=True)

with col2:
    fig, ax = plt.subplots(figsize=(4,3))
    ax.hist(df['Age'], bins=15, color='skyblue', edgecolor='black')
    ax.set_title("Age Distribution", fontsize=10)
    st.pyplot(fig, clear_figure=True)

with col3:
    fig, ax = plt.subplots(figsize=(4,3))
    ax.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c='green', alpha=0.6, s=40)
    ax.set_xlabel('Annual Income', fontsize=9)
    ax.set_ylabel('Spending Score', fontsize=9)
    ax.set_title("Income vs Spending Score", fontsize=10)
    st.pyplot(fig, clear_figure=True)

# Clustering Visualizations
st.header("Clustering Visualization (PCA)")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(5,4))
    palette_k = sns.color_palette("Set1", n_colors=df['KMeans_Cluster'].nunique())
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df['KMeans_Cluster'], palette=palette_k, s=40, ax=ax)
    ax.scatter(new_customer_pca[:,0], new_customer_pca[:,1], c='black', s=80, marker='X', label='New Customer')
    ax.set_title("KMeans Clusters", fontsize=10)
    ax.legend(title="Cluster", fontsize=8, title_fontsize=9)
    st.pyplot(fig, clear_figure=True)

with col2:
    fig, ax = plt.subplots(figsize=(5,4))
    unique_db = df['DBSCAN_Cluster'].unique()
    palette_d = sns.color_palette("Set2", n_colors=len(unique_db))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df['DBSCAN_Cluster'], palette=palette_d, s=40, ax=ax)
    ax.scatter(new_customer_pca[:,0], new_customer_pca[:,1], c='black', s=80, marker='X', label='New Customer')
    ax.set_title("DBSCAN Clusters", fontsize=10)
    ax.legend(title="Cluster", fontsize=8, title_fontsize=9)
    st.pyplot(fig, clear_figure=True)

# Clustering Metrics Comparison
st.header("Clustering Metrics Comparison")

# Compute metrics
kmeans_labels = df['KMeans_Cluster']
dbscan_labels = df['DBSCAN_Cluster']

sil_k = silhouette_score(X_scaled, kmeans_labels)
db_k = davies_bouldin_score(X_scaled, kmeans_labels)
ch_k = calinski_harabasz_score(X_scaled, kmeans_labels)

# DBSCAN metrics
try:
    sil_d = silhouette_score(X_scaled, dbscan_labels)
except:
    sil_d = None
try:
    db_d = davies_bouldin_score(X_scaled, dbscan_labels)
except:
    db_d = None
try:
    ch_d = calinski_harabasz_score(X_scaled, dbscan_labels)
except:
    ch_d = None

# Round metrics to 4 decimal places
metrics_dict = {
    "Metric": ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Score"],
    "KMeans": [round(sil_k,4), round(db_k,4), round(ch_k,4)],
    "DBSCAN": [
        round(sil_d,4) if sil_d is not None else "N/A",
        round(db_d,4) if db_d is not None else "N/A",
        round(ch_d,4) if ch_d is not None else "N/A"
    ]
}

metrics_df = pd.DataFrame(metrics_dict)

# Display table
st.table(metrics_df)

