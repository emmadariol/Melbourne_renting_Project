import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

# 1. Load Data Efficiently
# We use st.cache_data so we only load the heavy file once, not on every click.
@st.cache_data
def load_market_data():
    try:
        # Load the same artifacts file created by train.py
        artifacts = joblib.load("model_artifacts.pkl")
        df = artifacts["database"]
        return df
    except FileNotFoundError:
        st.error("Stats Error: 'model_artifacts.pkl' not found. Run train.py first.")
        return pd.DataFrame()

class MarketStats:
    def __init__(self):
        self.df = load_market_data()

    def show_dashboard(self, selected_region, selected_type):
        """
        Main function to display all stats widgets.
        """
        if self.df.empty:
            return

        st.markdown(f"### Market Trends: {selected_region}")
        
        # Filter data for relevance
        # We filter by Region to keep the comparison fair, and Property Type (House vs Unit)
        # because comparing House prices to Unit prices skews the data.
        filtered_df = self.df[
            (self.df['Regionname'] == selected_region) & 
            (self.df['Type'] == selected_type)
        ]

        if filtered_df.empty:
            st.warning("Not enough data to generate statistics for this selection.")
            return

        # --- KPI ROW ---
        avg_price = filtered_df['Price'].mean()
        med_price = filtered_df['Price'].median()
        count = len(filtered_df)
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Market Average", f"${avg_price:,.0f}")
        k2.metric("Market Median", f"${med_price:,.0f}")
        k3.metric("Properties Analyzed", count)
        
        st.divider()

        # --- CHART 1: Price Distribution (Histogram) ---
        # "Is my budget realistic?"
        st.markdown("#### Price Distribution")
        fig_hist = px.histogram(
            filtered_df, 
            x="Price", 
            nbins=30, 
            title=f"Price Range for '{selected_type}' in {selected_region}",
            color_discrete_sequence=["#FF4B4B"] # Streamlit Red
        )
        fig_hist.update_layout(bargap=0.1, template="plotly_white")
        st.plotly_chart(fig_hist, use_container_width=True)

        # --- CHART 2: Price vs. Size (Scatter) ---
        # "Am I paying for the house or the land?"
        st.markdown("#### Price vs. Land Size")
        # Filter out extreme outliers for a better chart (e.g., massive rural properties)
        clean_scatter = filtered_df[filtered_df['Landsize'] < 2000] 
        
        fig_scatter = px.scatter(
            clean_scatter, 
            x="Landsize", 
            y="Price", 
            color="Rooms", # Color by room count adds extra insight
            title=f"Price vs Landsize (under 2000sqm)",
            template="plotly_white",
            labels={"Landsize": "Land Size (sqm)", "Price": "Price ($)"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)