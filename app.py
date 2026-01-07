import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.portfolio_recommendation.portfolio_recommendation import recommend_portfolio

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---------------------------
# Load models
# ---------------------------
rf_model = joblib.load("../models/risk_profiling/random_forest_model.joblib")
kmeans_model = joblib.load("../models/customer_segmentation/kmeans_model.joblib")


st.set_page_config(page_title="RoboAdvisor AI", page_icon="🤖", layout="centered")

st.title("🤖 AI-Powered RoboAdvisor")
st.subheader("Get your personalized investment recommendation instantly!")


col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    income = st.number_input("Annual Income (₹)", min_value=100000, max_value=5000000, step=10000)
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, step=1)

with col2:
    savings_ratio = st.slider("Savings Ratio (Savings / Income)", 0.0, 1.0, 0.3, step=0.01)
    debt_ratio = st.slider("Debt Ratio (Debt / Income)", 0.0, 1.0, 0.2, step=0.01)
    assets = st.number_input("Total Assets (₹)", min_value=0, max_value=10, step=1)

knowledge_level = st.selectbox(
    "Financial Knowledge Level", 
    ["Low", "Medium", "High"]
)
goal = st.selectbox(
    "Investment Goal", 
    ["Wealth Creation", "Retirement", "Education", "Short-term Savings"]
)


knowledge_map = {"Low": 1, "Medium": 2, "High": 3}
goal_map = {
    "Wealth Creation": 1,
    "Retirement": 2,
    "Education": 3,
    "Short-term Savings": 4
}

knowledge_score = knowledge_map[knowledge_level]
goal_score = goal_map[goal]



if st.button("💼 Get Recommendation"):
    
    X_input = np.array([[age, income, dependents, savings_ratio, debt_ratio, 
                     assets, knowledge_score, goal_score]])


    risk_pred = rf_model.predict(X_input)[0]
    cluster_pred = kmeans_model.predict(X_input)[0]

    
    portfolio = recommend_portfolio(risk_pred, cluster_pred)

   
    st.success(f"Your predicted **risk level**: {risk_pred}")
    st.info(f"Your assigned **customer cluster**: {cluster_pred}")

    st.subheader("📊 Recommended Portfolio Allocation:")
    portfolio_df = pd.DataFrame(portfolio.items(), columns=["Asset", "Allocation (%)"])
    portfolio_df["Allocation (%)"] *= 100

    st.dataframe(portfolio_df, use_container_width=True)

    st.bar_chart(portfolio_df.set_index("Asset"))

    st.caption("This recommendation is generated using your financial profile and AI-based models.")