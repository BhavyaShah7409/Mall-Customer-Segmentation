# Mall Customers Segmentation (Clustering Demo)

This project segments mall customers into different groups based on their demographics and spending behavior.  
We use unsupervised learning techniques — **KMeans** and **DBSCAN** — to cluster customers and provide insights for:

- Targeted marketing campaigns  
- Personalized recommendations  
- Customer behavior analysis  

The app is built with **Streamlit** and allows adding a new customer to predict their cluster in real time.  

---

## Features
- Preview the **Mall Customers dataset**  
- Perform clustering with **KMeans** and **DBSCAN**  
- View clustering evaluation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)  
- Add a new customer through the sidebar to check cluster assignment  

---

## Project Structure
```
├── app.py                # Main Streamlit application
├── requirements.txt      # Project dependencies
├── models/
│   ├── kmeans_model.pkl  # Pre-trained KMeans model
│   └── dbscan_model.pkl  # Pre-trained DBSCAN model
├── data/
│   └── Mall_Customers.csv # Dataset
└── README.md             # Project documentation
```
