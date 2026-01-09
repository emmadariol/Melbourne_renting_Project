import streamlit as st
import requests
import pandas as pd
import pydeck as pdk
import os

# Use Docker network URL if available, otherwise fallback to localhost
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Melb House Finder", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Polish ---
st.markdown("""
<style>
    /* Use Streamlit's built-in CSS variables.
       These automatically change values depending on Light/Dark mode.
    */
    .stMetric {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid var(--background-color) !important;
        color: var(--text-color) !important;
        padding: 10px;
        border-radius: 5px;
    }
    
    /* Force the text inside the metric to respect the theme color */
    .stMetric [data-testid="stMetricValue"], 
    .stMetric [data-testid="stMetricLabel"] {
        color: var(--text-color) !important;
    }

    /* Adjust sidebar expander font size */
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

# --- Sidebar Inputs (Organized) ---
with st.sidebar:
    st.header("Search Filters")
    
    # Group 1: Essential
    with st.expander("Location & Budget", expanded=True):
        region = st.selectbox("Region", [
            "Southern Metropolitan", "Northern Metropolitan", "Western Metropolitan", 
            "Eastern Metropolitan", "South-Eastern Metropolitan", "Eastern Victoria",
            "Northern Victoria", "Western Victoria"
        ])
        price = st.number_input("Target Price ($)", 100_000, 10_000_000, 1_000_000, step=50_000)
        dist = st.slider("Max Distance (CBD)", 1.0, 50.0, 10.0, format="%d km")

    # Group 2: Property Specs
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

    # Group 3: Advanced
    with st.expander("Advanced & Type", expanded=False):
        prop_count = st.slider("Suburb Density (Props)", 0, 22000, 5000, 
                               help="Higher number = more dense suburb")
        
        type_options = {
            "h": "House/Cottage", "u": "Unit/Duplex", "t": "Townhouse",
            "br": "Bedroom(s)", "dev site": "Dev Site", "o res": "Other Res"
        }
        type_code = st.selectbox("Type", options=list(type_options.keys()), 
                                 format_func=lambda x: type_options[x])

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
    
    with st.spinner("Crunching numbers and locating properties..."):
        try:
            res = requests.post(f"{API_URL}/recommend", json=payload)
            if res.status_code == 200:
                st.session_state.recommendations = res.json().get("recommendations", [])
                st.session_state.search_performed = True
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
        # Create Tabs for different views
        tab_map, tab_list = st.tabs(["Map View", "Listings Details"])
        
        # --- TAB 1: Interactive Map ---
        with tab_map:
            # Prepare data for PyDeck
            map_df = pd.DataFrame(data)
            # Ensure coordinates are float
            map_df['lat'] = map_df['Lattitude'].astype(float)
            map_df['lon'] = map_df['Longtitude'].astype(float)
            
            # Define view state centered on the first result
            view_state = pdk.ViewState(
                latitude=map_df['lat'].mean(),
                longitude=map_df['lon'].mean(),
                zoom=11,
                pitch=0
            )

            # Define Layer with Tooltip capability
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=200,
                pickable=True
            )
            
            # Render Map with Tooltip
            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                initial_view_state=view_state,
                layers=[layer],
                tooltip={"text": "{Address}\nPrice: ${Price}\n{Rooms} Beds"}
            ))
            st.caption("Hover over points to see details.")

        # --- TAB 2: Detailed Listings ---
        with tab_list:
            for i, house in enumerate(data):
                # Create a card-like container
                with st.container(border=True):
                    # Header Row
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.subheader(f"{i+1}. {house.get('Address', 'Unknown Address')}")
                        st.caption(f"Suburb: **{house['Suburb']}** | Seller: {house['SellerG']}")
                    with c2:
                        st.metric("Price", f"${house['Price']:,.0f}")
                    
                    st.divider()
                    
                    # Icons Row
                    ic1, ic2, ic3, ic4 = st.columns(4)
                    ic1.markdown(f"**{house['Rooms']}** Beds")
                    ic2.markdown(f"**{house['Bathroom']}** Baths")
                    ic3.markdown(f"**{house['Car']}** Spots")
                    ic4.markdown(f"**{house['Landsize']}** m²")
                    
                    # Collapsible Neighborhood Report
                    with st.expander("Neighborhood Report (Amenities)"):
                        if st.button("Generate Infrastructure Scan", key=f"geo_{i}"):
                            with st.spinner("Scanning OSM Data..."):
                                try:
                                    report_res = requests.get(f"{API_URL}/house_report", 
                                                            params={"lat": house['Lattitude'], "lon": house['Longtitude']})
                                    report_data = report_res.json()

                                    if report_data["status"] == "success":
                                        rdata = report_data["data"]
                                        
                                        # Use a cleaner table-like layout
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
                                            
                                    else:
                                        st.error("Could not fetch location data.")
                                except Exception as e:
                                    st.error(f"Error fetching report: {e}")
                        else:
                            st.info("Click to scan the neighborhood for transport, schools, and parks.")