# Mall Customers Segmentation (Clustering Demo)

This project segments mall customers into different groups based on their demographics and spending behavior.  
We use unsupervised learning techniques — **KMeans** and **DBSCAN** — to cluster customers and provide insights for:

- Targeted marketing campaigns  
- Personalized recommendations  
- Customer behavior analysis  

The app is built with **Streamlit** and allows adding a new customer to predict their cluster in real time.  

---

## Features
- Upload or preview the **Mall Customers dataset**  
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

---

## Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/mall-customers-segmentation.git
   cd mall-customers-segmentation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Open the provided URL in your browser.

---

## Clustering Metrics Example
| Metric                 | KMeans  | DBSCAN  |
|-------------------------|---------|---------|
| Silhouette Score        | 0.2410  | 0.0938  |
| Davies-Bouldin Index    | 1.2540  | 1.5288  |
| Calinski-Harabasz Score | 52.9380 | 17.9151 |

---

## Requirements
- `streamlit`  
- `pandas`  
- `numpy`  
- `scikit-learn`  
- `joblib`  

(Optional: `matplotlib`, `seaborn` if you add EDA visualizations)  

---

## Deployment
You can deploy this project easily on:
- Streamlit Community Cloud  
- Heroku  
- Railway  

---

## License
This project is licensed under the MIT License.  
