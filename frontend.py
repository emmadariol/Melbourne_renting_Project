import streamlit as st
import requests
import pandas as pd
import pydeck as pdk
import os
import stats  # Ensure stats.py is in the same folder

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
    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        transition: border 0.3s;
    }
    /* Compact Color Picker */
    div[data-testid="stColorPicker"] > label {
        font-size: 0.8rem;
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
if "favorites" not in st.session_state:
    st.session_state.favorites = []

# --- Helper: Convert Hex to RGBA for Pydeck ---
def hex_to_rgba(hex_code, alpha=200):
    hex_code = hex_code.lstrip('#')
    return [int(hex_code[i:i+2], 16) for i in (0, 2, 4)] + [alpha]

# --- Helper Function: Toggle Favorite ---
def toggle_favorite(house):
    # Check if existing (match by Address)
    existing_idx = next((i for i, h in enumerate(st.session_state.favorites) if h['Address'] == house['Address']), -1)
    
    if existing_idx != -1:
        # Remove
        st.session_state.favorites.pop(existing_idx)
        st.toast(f"Removed {house['Address']} from favorites", icon="🗑️")
    else:
        # Add with a default color (Gold)
        house_data = house.copy()
        house_data['color'] = '#FFD700' # Default Gold
        st.session_state.favorites.append(house_data)
        st.toast(f"Added {house['Address']} to favorites!", icon="❤️")

# --- Helper Function: Render House Card ---
# Updated to accept 'radius_meters' and display FULL scan results
def render_house_card(house, index, radius_meters, is_selected=False, key_suffix=""):
    card_title = f"🎯 Selected: {house.get('Address', 'Unknown')}" if is_selected else f"{index+1}. {house.get('Address', 'Unknown')}"
    
    # Check status
    is_fav = any(h['Address'] == house['Address'] for h in st.session_state.favorites)
    
    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.subheader(card_title)
            st.caption(f"Suburb: **{house['Suburb']}** | Seller: {house['SellerG']}")
        with c2:
            st.metric("Price", f"${house['Price']:,.0f}")
        
        st.divider()
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.markdown(f"🛏️ **{house['Rooms']}** Beds")
        ic2.markdown(f"🛁 **{house['Bathroom']}** Baths")
        ic3.markdown(f"🚗 **{house['Car']}** Spots")
        ic4.markdown(f"📐 **{house['Landsize']}** m²")
        
        st.divider()
        ac1, ac2, ac3 = st.columns([1, 1, 1])
        with ac1:
            if not is_selected:
                if st.button(f"📍 Locate", key=f"btn_loc_{index}_{key_suffix}", use_container_width=True):
                    st.session_state.selected_idx = index
                    st.rerun()
            else:
                st.markdown("✅ *Selected*")

        with ac2:
            fav_label = "❤️ Unpin" if is_fav else "🤍 Pin"
            st.button(fav_label, key=f"btn_fav_{index}_{key_suffix}", on_click=toggle_favorite, args=(house,), use_container_width=True)

        with ac3:
            # Generate Scan Button
            if st.button("Generate Scan", key=f"btn_scan_{index}_{key_suffix}", use_container_width=True):
                with st.spinner(f"Scanning within {radius_meters}m..."):
                    try:
                        # PASS RADIUS TO API
                        report_res = requests.get(f"{API_URL}/house_report", 
                                                params={
                                                    "lat": house['Lattitude'], 
                                                    "lon": house['Longtitude'],
                                                    "radius": radius_meters
                                                })
                        if report_res.status_code == 200:
                            rdata = report_res.json().get("data", {})
                            st.session_state[f"scan_{index}"] = rdata
                        else:
                            st.error("Scan Failed")
                    except Exception as e:
                        st.error(f"Error: {e}")
            
        # --- DISPLAY SCAN RESULTS (Full View) ---
        if f"scan_{index}" in st.session_state:
            rdata = st.session_state[f"scan_{index}"]
            
            st.markdown("---")
            st.markdown(f"##### 📍 Neighborhood Connectivity ({radius_meters}m Radius)")
            
            sc1, sc2, sc3 = st.columns(3)
            
            with sc1:
                st.caption("🚆 **Transport**")
                st.write(f"Train: **{rdata.get('train_station', 'N/A')}m**")
                st.write(f"Tram: **{rdata.get('tram_stop', 'N/A')}m**")
            
            with sc2:
                st.caption("🎓 **Services**")
                st.write(f"School: **{rdata.get('school', 'N/A')}m**")
                st.write(f"Shop: **{rdata.get('supermarket', 'N/A')}m**")

            with sc3:
                st.caption("🌳 **Lifestyle**")
                st.write(f"Park: **{rdata.get('park', 'N/A')}m**")
                st.write(f"Cafe: **{rdata.get('cafe', 'N/A')}m**")

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
        
        # NEW: Scan Radius Slider
        search_radius = st.slider("Amenity Scan Radius (m)", 500, 5000, 1000, step=100, 
                                  help="How far to look for trains, parks, and cafes.")

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

    # --- FAVORITES SECTION (Personalized Colors) ---
    st.divider()
    st.header(f"❤️ Saved Homes ({len(st.session_state.favorites)})")
    
    if st.session_state.favorites:
        for i, fav in enumerate(st.session_state.favorites):
            with st.container(border=True):
                st.write(f"**{fav['Address']}**")
                
                # --- INDIVIDUAL COLOR PICKER ---
                current_color = fav.get('color', '#FFD700')
                
                col_pick, col_del = st.columns([4, 1])
                with col_pick:
                    new_color = st.color_picker(
                        "Pin Color", 
                        value=current_color, 
                        key=f"fav_col_{i}",
                        label_visibility="collapsed"
                    )
                    
                    if new_color != current_color:
                        st.session_state.favorites[i]['color'] = new_color

                with col_del:
                    if st.button("🗑️", key=f"del_fav_{i}"):
                        toggle_favorite(fav)
                        st.rerun()
    else:
        st.caption("No favorites yet.")

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
        tab_map, tab_stats = st.tabs(["🏡 Map & Listings", "📊 Market Insights"])
        
        # --- TAB 1: Map & Listings ---
        with tab_map:
            map_df = pd.DataFrame(data)
            map_df['lat'] = map_df['Lattitude'].astype(float)
            map_df['lon'] = map_df['Longtitude'].astype(float)
            
            # --- COLOR LOGIC ---
            colors = []
            for idx, row in map_df.iterrows():
                house_addr = data[idx]['Address']
                
                # Check for personalized favorites
                fav_obj = next((f for f in st.session_state.favorites if f['Address'] == house_addr), None)
                
                if idx == st.session_state.selected_idx:
                    colors.append([0, 200, 0, 255]) # GREEN (Selected)
                elif fav_obj:
                    fav_color = fav_obj.get('color', '#FFD700')
                    colors.append(hex_to_rgba(fav_color, alpha=255)) # USER COLOR
                else:
                    colors.append([200, 30, 0, 160]) # RED (Default)
            
            map_df['color'] = colors

            # Center Map
            if st.session_state.selected_idx is not None and st.session_state.selected_idx < len(map_df):
                idx = st.session_state.selected_idx
                target_lat = map_df.at[idx, 'lat']
                target_lon = map_df.at[idx, 'lon']
                zoom_level = 14
            else:
                target_lat = map_df['lat'].mean()
                target_lon = map_df['lon'].mean()
                zoom_level = 11

            view_state = pdk.ViewState(latitude=target_lat, longitude=target_lon, zoom=zoom_level, pitch=0)

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position='[lon, lat]',
                get_fill_color='color',
                get_radius=200,
                pickable=True,
                auto_highlight=True,
                id="house-layer"
            )

            tooltip = {"html": "<b>{Address}</b><br/>${Price}", "style": {"backgroundColor": "#1f1f1f", "color": "white"}}

            map_event = st.pydeck_chart(
                pdk.Deck(map_style=None, initial_view_state=view_state, layers=[layer], tooltip=tooltip),
                on_select="rerun", selection_mode="single-object", key="main_map_chart" 
            )

            # Map Click Logic
            if map_event.selection and "indices" in map_event.selection:
                raw_indices = map_event.selection["indices"]
                safe_indices = []
                if isinstance(raw_indices, dict):
                    for idx_list in raw_indices.values():
                        safe_indices.extend(idx_list)
                else:
                    safe_indices = list(raw_indices)
                if safe_indices:
                    st.session_state.selected_idx = int(safe_indices[0])

            # Display Cards
            st.divider()
            # A. SHOW SELECTED HOUSE
            if st.session_state.selected_idx is not None:
                if st.session_state.selected_idx < len(data):
                    render_house_card(
                        data[st.session_state.selected_idx], 
                        st.session_state.selected_idx, 
                        radius_meters=search_radius,  # <--- Passing the slider value
                        is_selected=True, 
                        key_suffix="sel"
                    )
                    st.markdown("### 🏘️ Other Recommendations")
            
            # B. SHOW THE REST
            for i, house in enumerate(data):
                if i != st.session_state.selected_idx:
                    render_house_card(
                        house, 
                        i, 
                        radius_meters=search_radius, # <--- Passing the slider value
                        is_selected=False, 
                        key_suffix="list"
                    )

        with tab_stats:
            market_engine = stats.MarketStats()
            market_engine.show_dashboard(region, type_code)