import streamlit as st
import pandas as pd
import plotly.express as px
from model_manager import ModelManager # <--- Import

@st.cache_data
def load_market_data():
    try:
        manager = ModelManager()
        # This will now always fetch the file pointed to by registry.json
        artifacts = manager.load_current_model()
        df = artifacts["database"]
        return df
    except Exception as e:
        st.error(f"Stats Error: Could not load model. {e}")
        return pd.DataFrame()

class MarketStats:
    def __init__(self):
        self.df = load_market_data()

    def show_dashboard(self, selected_region, selected_type):
        """Main function to display all stats widgets."""
        if self.df.empty:
            return

        st.markdown(f"### Market Trends: {selected_region}")
        
        filtered_df = self.df[
            (self.df['Regionname'] == selected_region) & 
            (self.df['Type'] == selected_type)
        ]

        if filtered_df.empty:
            st.warning("Not enough data to generate statistics for this selection.")
            return

        # KPI ROW
        avg_price = filtered_df['Price'].mean()
        med_price = filtered_df['Price'].median()
        count = len(filtered_df)
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Market Average", f"${avg_price:,.0f}")
        k2.metric("Market Median", f"${med_price:,.0f}")
        k3.metric("Properties Analyzed", count)
        
        st.divider()

        # CHART 1: Price Distribution
        st.markdown("#### Price Distribution")
        fig_hist = px.histogram(
            filtered_df, x="Price", nbins=30, 
            title=f"Price Range for '{selected_type}' in {selected_region}",
            color_discrete_sequence=["#FF4B4B"]
        )
        fig_hist.update_layout(bargap=0.1, template="plotly_white")
        st.plotly_chart(fig_hist, use_container_width=True)

        # CHART 2: Scatter
        st.markdown("#### Price vs. Land Size")
        clean_scatter = filtered_df[filtered_df['Landsize'] < 2000] 
        
        fig_scatter = px.scatter(
            clean_scatter, x="Landsize", y="Price", color="Rooms",
            title=f"Price vs Landsize (under 2000sqm)",
            template="plotly_white",
            labels={"Landsize": "Land Size (sqm)", "Price": "Price ($)"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)