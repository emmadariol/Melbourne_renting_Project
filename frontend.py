import streamlit as st
import requests
import pandas as pd
import pydeck as pdk
import os
import stats

# Use Docker network URL if available, otherwise fallback to localhost
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Melb House Finder", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .stMetric {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid var(--background-color) !important;
        color: var(--text-color) !important;
        padding: 10px;
        border-radius: 5px;
    }
    .stMetric [data-testid="stMetricValue"], 
    .stMetric [data-testid="stMetricLabel"] {
        color: var(--text-color) !important;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("Melbourne Housing Recommender")
st.markdown("Find your dream home based on similarity to your ideal criteria.")

# --- Initialize Session State ---
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "search_performed" not in st.session_state:
    st.session_state.search_performed = False
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = None

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Search Filters")
    
    with st.expander("Location & Budget", expanded=True):
        region = st.selectbox("Region", [
            "Southern Metropolitan", "Northern Metropolitan", "Western Metropolitan", 
            "Eastern Metropolitan", "South-Eastern Metropolitan", "Eastern Victoria",
            "Northern Victoria", "Western Victoria"
        ])
        price = st.number_input("Target Price ($)", 100_000, 10_000_000, 1_000_000, step=50_000)
        dist = st.slider("Max Distance (CBD)", 1.0, 50.0, 10.0, format="%d km")

    with st.expander("Property Features", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            rooms = st.number_input("Rooms", 1, 10, 3)
            bathroom = st.number_input("Baths", 1, 5, 2)
        with col2:
            car = st.number_input("Cars", 0, 5, 1)
            year_built = st.number_input("Year", 1800, 2024, 2000)
            
        land = st.slider("Landsize (sqm)", 0, 10000, 600)
        build_area = st.slider("Building Area (sqm)", 0, 5000, 150)

    with st.expander("Advanced & Type", expanded=False):
        prop_count = st.slider("Suburb Density (Props)", 0, 22000, 5000)
        type_options = {
            "h": "House/Cottage", "u": "Unit/Duplex", "t": "Townhouse",
            "br": "Bedroom(s)", "dev site": "Dev Site", "o res": "Other Res"
        }
        type_code = st.selectbox("Type", options=list(type_options.keys()), 
                                 format_func=lambda x: type_options[x])

    st.divider()
    
    def reset_search_state():
        st.session_state.selected_idx = None
        
    search = st.button("Find Matches", type="primary", use_container_width=True, on_click=reset_search_state)

# --- Logic to Fetch Data ---
if search:
    payload = {
        "Rooms": rooms, "Price": price, "Distance": dist, "Bathroom": bathroom,
        "Car": car, "Landsize": land, "BuildingArea": build_area,
        "Propertycount": prop_count, "YearBuilt": year_built,
        "Type": type_code, "Regionname": region
    }
    
    with st.spinner("Crunching numbers..."):
        try:
            res = requests.post(f"{API_URL}/recommend", json=payload)
            if res.status_code == 200:
                st.session_state.recommendations = res.json().get("recommendations", [])
                st.session_state.search_performed = True
                st.session_state.selected_idx = None
            else:
                st.error(f"Error {res.status_code}: {res.text}")
        except Exception as e:
            st.error(f"Connection Error: {e}")

# --- Main Page Display ---
if st.session_state.search_performed:
    data = st.session_state.recommendations
    
    if not data:
        st.warning("No matches found. Try adjusting your filters.")
    else:
        tab_map, tab_list, tab_stats = st.tabs(["Map View", "Listings Details", "Market Insights"])
        
        # --- TAB 1: Interactive Map ---
        with tab_map:
            map_df = pd.DataFrame(data)
            map_df['lat'] = map_df['Lattitude'].astype(float)
            map_df['lon'] = map_df['Longtitude'].astype(float)

            view_state = pdk.ViewState(
                latitude=map_df['lat'].mean(),
                longitude=map_df['lon'].mean(),
                zoom=11, 
                pitch=0
            )

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=200,
                pickable=True,
                auto_highlight=True,
                id="house-layer"
            )

            tooltip = {
                "html": "<b>{Address}</b><br/>${Price}",
                "style": {"backgroundColor": "#1f1f1f", "color": "white"}
            }

            map_event = st.pydeck_chart(
                pdk.Deck(
                    map_style=None,
                    initial_view_state=view_state,
                    layers=[layer],
                    tooltip=tooltip
                ),
                on_select="rerun",
                selection_mode="single-object",
                key="main_map_chart" 
            )

            # --- FIXED SELECTION LOGIC ---
            if map_event.selection and "indices" in map_event.selection:
                raw_indices = map_event.selection["indices"]
                
                # Logic to handle Dictionary (Layer ID) vs List (Flat indices)
                safe_indices = []
                if isinstance(raw_indices, dict):
                    # It's a dict like {'house-layer': [0, ...]} -> Extract values
                    for idx_list in raw_indices.values():
                        safe_indices.extend(idx_list)
                else:
                    # It's already a flat list like [0, ...]
                    safe_indices = list(raw_indices)
                
                # If we found any valid indices, update the state
                if safe_indices:
                    st.session_state.selected_idx = int(safe_indices[0])
            
            # --- DISPLAY SELECTED HOUSE ---
            if st.session_state.selected_idx is not None:
                if st.session_state.selected_idx < len(data):
                    sel_house = data[st.session_state.selected_idx]
                    
                    st.divider()
                    st.markdown(f"### {sel_house.get('Address', 'Unknown')}")
                    
                    with st.container(border=True):
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.caption(f"Suburb: **{sel_house['Suburb']}** | Seller: {sel_house['SellerG']}")
                        with c2:
                            st.metric("Price", f"${sel_house['Price']:,.0f}")
                        
                        ic1, ic2, ic3, ic4 = st.columns(4)
                        ic1.markdown(f"🛏️ **{sel_house['Rooms']}** Beds")
                        ic2.markdown(f"🛁 **{sel_house['Bathroom']}** Baths")
                        ic3.markdown(f"🚗 **{sel_house['Car']}** Spots")
                        ic4.markdown(f"📐 **{sel_house['Landsize']}** m²")
                        
                        if st.button("Generate Infrastructure Scan", key=f"geo_select_btn_fixed"):
                            with st.spinner("Scanning neighborhood..."):
                                try:
                                    report_res = requests.get(f"{API_URL}/house_report", 
                                                            params={"lat": sel_house['Lattitude'], "lon": sel_house['Longtitude']})
                                    if report_res.status_code == 200:
                                        rdata = report_res.json()["data"]
                                        rc1, rc2 = st.columns(2)
                                        with rc1:
                                            st.markdown("##### 🚆 Transport")
                                            st.write(f"• Train: **{rdata.get('train_station') or 'N/A'}m**")
                                            st.write(f"• Tram: **{rdata.get('tram_stop') or 'N/A'}m**")
                                        with rc2:
                                            st.markdown("##### 🌳 Lifestyle")
                                            st.write(f"• Park: **{rdata.get('park') or 'N/A'}m**")
                                            st.write(f"• Cafe: **{rdata.get('cafe') or 'N/A'}m**")
                                    else:
                                        st.error("Scan failed.")
                                except Exception as e:
                                    st.error(f"Error: {e}")
                else:
                    st.session_state.selected_idx = None
            else:
                st.info("Click on any red dot on the map above to see property details here.")

        # --- TAB 2: Listings ---
        with tab_list:
            for i, house in enumerate(data):
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.subheader(f"{i+1}. {house.get('Address', 'Unknown')}")
                        st.caption(f"Suburb: **{house['Suburb']}**")
                    with c2:
                        st.metric("Price", f"${house['Price']:,.0f}")
                    st.divider()
                    st.write(f"**{house['Rooms']}** Beds | **{house['Bathroom']}** Baths | **{house['Car']}** Cars")

        # --- TAB 3: Stats ---
        with tab_stats:
            market_engine = stats.MarketStats()
            market_engine.show_dashboard(region, type_code)