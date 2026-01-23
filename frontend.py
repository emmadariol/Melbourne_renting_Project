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

# --- Custom CSS for Visibility (Light & Dark Mode Support) ---
st.markdown("""
<style>
    /* 1. Metric Box Styling */
    .stMetric {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid var(--background-color) !important;
        color: var(--text-color) !important;
        padding: 10px;
        border-radius: 5px;
    }
    
    /* 2. Text Visibility Fix */
    .stMetric [data-testid="stMetricValue"], 
    .stMetric [data-testid="stMetricLabel"] {
        color: var(--text-color) !important;
    }

    /* 3. Sidebar Font Adjustment */
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
if "geo_reports" not in st.session_state:
    st.session_state.geo_reports = {}

# --- Helper Function to Render a House Card ---
def render_house_card(index, house, is_highlighted=False):
    """Renders the details of a single house."""
    # Add a visual border or highlight if it's the selected one
    container_border = True
    
    with st.container(border=container_border):
        if is_highlighted:
            st.markdown("### 🎯 Selected Property")
        
        # Header
        c1, c2 = st.columns([3, 1])
        with c1:
            st.subheader(f"{index+1}. {house.get('Address', 'Unknown Address')}")
            st.caption(f"Suburb: **{house['Suburb']}** | Seller: {house['SellerG']}")
        with c2:
            st.metric("Price", f"${house['Price']:,.0f}")
        
        st.divider()
        
        # Icons
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.markdown(f"**{house['Rooms']}** Beds")
        ic2.markdown(f"**{house['Bathroom']}** Baths")
        ic3.markdown(f"**{house['Car']}** Spots")
        ic4.markdown(f"**{house['Landsize']}** m²")
        
        # Neighborhood Report
        with st.expander("Neighborhood Report (Amenities)", expanded=is_highlighted):
            cache_key = index
            rdata = st.session_state.geo_reports.get(cache_key)

            if not rdata:
                if st.button("Generate Infrastructure Scan", key=f"geo_{index}"):
                    with st.spinner("Scanning OSM Data..."):
                        try:
                            report_res = requests.get(f"{API_URL}/house_report", 
                                                    params={"lat": house['Lattitude'], "lon": house['Longtitude']})
                            report_data = report_res.json()

                            if report_data["status"] == "success":
                                rdata = report_data["data"]
                                st.session_state.geo_reports[cache_key] = rdata
                                st.rerun() # Rerun to show data immediately
                            else:
                                st.error("Could not fetch location data.")
                        except Exception as e:
                            st.error(f"Error fetching report: {e}")
                else:
                    st.info("Click to scan the neighborhood for transport, schools, and parks.")
            
            if rdata:
                rc1, rc2 = st.columns(2)
                with rc1:
                    st.markdown("##### Transport")
                    st.write(f"• Train: **{rdata.get('train_station') or 'N/A'}m**")
                    st.write(f"• Tram: **{rdata.get('tram_stop') or 'N/A'}m**")
                with rc2:
                    st.markdown("##### Lifestyle")
                    st.write(f"• Park: **{rdata.get('park') or 'N/A'}m**")
                    st.write(f"• Cafe: **{rdata.get('cafe') or 'N/A'}m**")
                    st.write(f"• School: **{rdata.get('school') or 'N/A'}m**")

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
        prop_count = st.slider("Suburb Density (Props)", 0, 22000, 5000, help="Higher number = more dense suburb")
        type_options = {
            "h": "House/Cottage", "u": "Unit/Duplex", "t": "Townhouse",
            "br": "Bedroom(s)", "dev site": "Dev Site", "o res": "Other Res"
        }
        type_code = st.selectbox("Type", options=list(type_options.keys()), format_func=lambda x: type_options[x])
    
    with st.expander("Required Nearby Amenities", expanded=False):
        st.markdown("**Select services that must be nearby:**")
        st.caption("Each service has its own sensible search radius")
        amenity_options = {
            "supermarket": "🛒 Supermarket (500m)", "bus_stop": "🚌 Bus Stop (300m)",
            "tram_stop": "🚊 Tram Stop (400m)", "train_station": "🚆 Train Station (1.5km)",
            "school": "🏫 School (1.5km)", "park": "🌳 Park (1km)",
            "cafe": "☕ Cafe (500m)", "gym": "💪 Gym (1km)",
            "hospital": "🏥 Hospital (3km)", "lake": "🌊 Lake (3km)"
        }
        selected_amenities = []
        for key, label in amenity_options.items():
            if st.checkbox(label, key=f"amenity_{key}"):
                selected_amenities.append(key)

    st.divider()
    search = st.button("Find Matches", type="primary", use_container_width=True)

# --- Logic to Fetch Data ---
if search:
    payload = {
        "Rooms": rooms, "Price": price, "Distance": dist, "Bathroom": bathroom,
        "Car": car, "Landsize": land, "BuildingArea": build_area,
        "Propertycount": prop_count, "YearBuilt": year_built,
        "Type": type_code, "Regionname": region
    }
    if selected_amenities:
        payload["required_amenities"] = selected_amenities
    
    with st.spinner("Crunching numbers and locating properties..."):
        try:
            res = requests.post(f"{API_URL}/recommend", json=payload)
            if res.status_code == 200:
                st.session_state.recommendations = res.json().get("recommendations", [])
                st.session_state.search_performed = True
                st.session_state.geo_reports = {}
                if selected_amenities:
                    st.info(f"Filtered results to include only houses near: {', '.join(selected_amenities)}")
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
        # [UPDATED] Using Tabs to separate "Search & Details" from "Market Insights"
        tab_main, tab_stats = st.tabs(["Property Search & Details", "📊 Market Insights"])

        # --- TAB 1: Search, Map, and Listings ---
        with tab_main:
            # 1. MAP VIEW
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
                id="house_layer",
                auto_highlight=True
            )
            
            tooltip_config = {
                "html": "<b>{Address}</b><br/>Price: ${Price}<br/>{Rooms} Beds",
                "style": {
                    "backgroundColor": "#1f1f1f",
                    "color": "white",
                    "padding": "10px",
                    "borderRadius": "5px"
                }
            }

            # ON_SELECT logic
            event = st.pydeck_chart(pdk.Deck(
                map_style = None,
                initial_view_state=view_state,
                layers=[layer],
                tooltip=tooltip_config
            ), on_select="rerun", selection_mode="single-object")
            
            st.caption("Click on a red dot to view property details below.")
            
            # 2. DETERMINE SELECTED HOUSE
            selected_index = None
            if event.selection and "indices" in event.selection and "house_layer" in event.selection["indices"]:
                indices = event.selection["indices"]["house_layer"]
                if indices:
                    selected_index = indices[0]

            # 3. DISPLAY SELECTED HOUSE
            if selected_index is not None:
                st.divider()
                render_house_card(selected_index, data[selected_index], is_highlighted=True)
                
            # 4. DISPLAY OTHER RECOMMENDATIONS
            st.divider()
            st.subheader("All Recommended Properties")
            
            for i, house in enumerate(data):
                # Skip the one currently shown in the "Selected" section
                if i == selected_index:
                    continue
                render_house_card(i, house)

        # --- TAB 2: Market Insights ---
        with tab_stats:
            market_engine = stats.MarketStats()
            market_engine.show_dashboard(region, type_code)