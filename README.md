# Machine Learning Robo-Advisory Platform

A comprehensive financial advisory and portfolio optimization system driven by machine learning. This platform models market data patterns and client metrics to generate automated, data-driven investment strategies, featuring a fully interactive frontend.

## 🧠 Algorithmic Implementation

The core intelligence of the advisory system is built on a multi-model machine learning architecture utilizing Scikit-learn:
* **Risk Profiling (K-Means Clustering):** Ingests multi-dimensional client data to cluster investors into distinct, optimized risk tolerance profiles.
* **Market Trend Prediction (Random Forest):** Employs an ensemble learning method utilizing decision trees to analyze historical market data and predict directional asset movements with high accuracy.
* **Outcome Optimization (Logistic Regression):** Utilizes binary classification to model the probability of specific portfolio outcomes based on historical volatility and return metrics.

## 🛠️ Tech Stack
* **Backend & ML:** Python, Scikit-learn, Pandas, NumPy
* **Frontend:** Streamlit 

## 📁 Repository Structure

* `/data` — Contains the raw and processed datasets used for training the clustering and classification models.
* `/data_preprocessing` — Modules dedicated to data cleaning, feature engineering, and scaling.
* `/models` — Stores the serialized, pre-trained Scikit-learn models for rapid inference by the frontend.
* `/src` — Core source code containing the backend business logic and machine learning pipelines.
* `app.py` — The main Streamlit application script that drives the interactive dashboard.
* `data_generation.ipynb` — Jupyter notebook detailing the exploratory data analysis and the generation/simulation of the foundational datasets.

## 🚀 Installation & Setup

1. Clone the repository:
```bash
   git clone [https://github.com/Shivaay-Bajoria/RoboAdvisory-System.git](https://github.com/Shivaay-Bajoria/RoboAdvisory-System.git)
   cd RoboAdvisory-System
```

2. Install the necessary data science and frontend dependencies:
```bash
   pip install streamlit scikit-learn pandas numpy
```
3. Launch the Streamlit application:
```bash
   streamlit run app.py
```
Access the dashboard via your local browser at http://localhost:8501.
