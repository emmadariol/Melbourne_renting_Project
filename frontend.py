import streamlit as st
import requests
import pandas as pd
import pydeck as pdk
import os
import stats

# Default to Flask port
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Melb House Finder", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS ---
st.markdown("""
<style>
    .stMetric { background-color: var(--secondary-background-color) !important; border: 1px solid var(--background-color) !important; padding: 10px; border-radius: 5px; }
    div[data-testid="stSidebar"] div.stSelectbox:first-of-type { border-bottom: 2px solid #FF4B4B; padding-bottom: 20px; margin-bottom: 20px; }
    div[data-testid="stVerticalBlock"] > div { transition: all 0.5s ease; }
</style>
""", unsafe_allow_html=True)

st.title("Melbourne Housing System")

# --- Session State ---
if "recommendations" not in st.session_state: st.session_state.recommendations = []
if "geo_reports" not in st.session_state: st.session_state.geo_reports = []
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if "selected_idx" not in st.session_state: st.session_state.selected_idx = None
if "edit_house_data" not in st.session_state: st.session_state.edit_house_data = None
if "verified_address" not in st.session_state: st.session_state.verified_address = None
if "address_verified" not in st.session_state: st.session_state.address_verified = False

# --- CONSTANTS ---
COUNCILS = [
    "Yarra City Council", "Moonee Valley City Council", "Port Phillip City Council", 
    "Darebin City Council", "Hume City Council", "Maribyrnong City Council", 
    "Boroondara City Council", "Moreland City Council", "Stonnington City Council", 
    "Banyule City Council", "Melbourne City Council", "Bayside City Council", 
    "Glen Eira City Council", "Monash City Council", "Whitehorse City Council", 
    "Manningham City Council", "Kingston City Council", "Brimbank City Council", 
    "Hobsons Bay City Council", "Wyndham City Council", "Frankston City Council", 
    "Knox City Council", "Maroondah City Council", "Greater Dandenong City Council", 
    "Casey City Council", "Whittlesea City Council", "Melton City Council", 
    "Nillumbik Shire Council", "Yarra Ranges Shire Council", "Macedon Ranges Shire Council"
]

# --- Helpers ---
def render_amenity_report(data):
    if not data:
        st.warning("No infrastructure data available.")
        return
    categories = {
        "🚍 Transport": {"train_station": "Train Station", "tram_stop": "Tram Stop", "bus_stop": "Bus Stop"},
        "🛒 Services": {"supermarket": "Supermarket", "school": "School", "hospital": "Hospital"},
        "🌳 Lifestyle": {"cafe": "Cafe", "park": "Park", "gym": "Gym", "lake": "Lake"}
    }
    for category_name, items in categories.items():
        st.caption(f"**{category_name}**")
        cols = st.columns(len(items))
        for idx, (key, label) in enumerate(items.items()):
            distance = data.get(key)
            with cols[idx]:
                if distance is not None:
                    delta = "Close" if distance < 500 else ("Medium" if distance < 1500 else None)
                    color = "normal" if distance < 500 else "off"
                    st.metric(label=label, value=f"{int(distance)} m", delta=delta, delta_color=color)
                else:
                    st.metric(label=label, value="--")
        st.divider()

def safe_val(val, default_type=float, default_val=0):
    if val is None or pd.isna(val): return default_type(default_val)
    try: return default_type(val)
    except: return default_type(default_val)

# ==========================================
# 1. SIDEBAR
# ==========================================
with st.sidebar:
    st.header("🔑 Access Control")
    role = st.selectbox("Select User Role", ["Home Seeker", "Real Estate Agent"])
    st.divider()

    if role == "Real Estate Agent":
        if not st.session_state.authenticated:
            st.warning("Restricted Access")
            password = st.text_input("Enter Agent Password", type="password")
            if st.button("Login"):
                if password == "admin":
                    st.session_state.authenticated = True
                    st.rerun()
                else: st.error("Incorrect password")
            st.stop()
        else:
            st.success("Logged in as Agent")
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.edit_house_data = None
                st.rerun()

# ==========================================
# 2. HOME SEEKER VIEW
# ==========================================
if role == "Home Seeker":
    st.markdown("### 🏡 Find your ideal home")
    
    with st.sidebar:
        st.header("Search Filters")
        with st.expander("Location & Budget", expanded=True):
            region = st.selectbox("Region", [
                "Southern Metropolitan", "Northern Metropolitan", "Western Metropolitan", 
                "Eastern Metropolitan", "South-Eastern Metropolitan", "Eastern Victoria",
                "Northern Victoria", "Western Victoria"
            ])
            price = st.number_input("Target Price ($)", 100_000, 10_000_000, 1_000_000, step=50_000)
            dist = st.slider("Max Distance (CBD)", 1.0, 50.0, 10.0)

        with st.expander("Property Features", expanded=True):
            c1, c2 = st.columns(2)
            rooms = c1.number_input("Rooms", 1, 10, 3)
            bathroom = c2.number_input("Baths", 1, 5, 2)
            car = c1.number_input("Cars", 0, 5, 1)
            year_built = c2.number_input("Year", 1800, 2024, 2000)
            land = st.slider("Land (sqm)", 0, 10000, 600)
            build_area = st.slider("Build Area (sqm)", 0, 5000, 150)

        type_map = {"h": "House", "u": "Unit", "t": "Townhouse"}
        type_code = st.selectbox("Type", options=list(type_map.keys()), format_func=lambda x: type_map[x])
        prop_count = 4000.0 

        search = st.button("Find Matches", type="primary", use_container_width=True)

    if search:
        payload = {
            "Rooms": rooms, "Price": price, "Distance": dist, "Bathroom": bathroom,
            "Car": car, "Landsize": land, "BuildingArea": build_area,
            "Propertycount": prop_count, "YearBuilt": year_built,
            "Type": type_code, "Regionname": region,
            "CouncilArea": "Yarra City Council" 
        }
        with st.spinner("Searching active listings..."):
            try:
                res = requests.post(f"{API_URL}/recommend", json=payload)
                if res.status_code == 200:
                    st.session_state.recommendations = res.json().get("recommendations", [])
                    st.session_state.selected_idx = None 
                else: st.error(f"Error: {res.text}")
            except Exception as e: st.error(f"Connection Failed: {e}")

    data = st.session_state.recommendations
    if not data:
        st.info("Use the sidebar filters to search for active rental properties.")
    else:
        st.caption("📍 **Click a red dot** or use the **View on Map** buttons below to highlight a property.")
        
        # --- ENHANCED MAP LOGIC ---
        map_df = pd.DataFrame(data).reset_index() # reset_index allows the layer to see 'index'
        selected_idx = st.session_state.get("selected_idx")

        # Define conditional logic for the selected dot: yellow and larger if index matches
        color_exp = f"index == {selected_idx} ? [255, 255, 0, 255] : [200, 30, 0, 160]"
        radius_exp = f"index == {selected_idx} ? 600 : 300"

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position='[Longtitude, Lattitude]',
            get_color=color_exp,
            get_radius=radius_exp,
            pickable=True,
            id="house_layer",
            auto_highlight=True,
            # update_triggers ensures the map redraws when selected_idx changes
            update_triggers={'get_color': [selected_idx], 'get_radius': [selected_idx]}
        )

        view_state = pdk.ViewState(latitude=map_df['Lattitude'].mean(), longitude=map_df['Longtitude'].mean(), zoom=11)
        selection = st.pydeck_chart(
            pdk.Deck(initial_view_state=view_state, layers=[layer], tooltip={"html": "<b>{Address}</b><br>${Price}"}), 
            on_select="rerun", 
            selection_mode="single-object"
        )
        
        # Update session state from map click
        if selection.selection and "indices" in selection.selection and "house_layer" in selection.selection["indices"]:
            indices = selection.selection["indices"]["house_layer"]
            if indices:
                st.session_state.selected_idx = indices[0]

        # Process data for display (bringing selected item to top)
        display_data = data.copy()
        is_reordered = False
        selected_item_data = None
        
        if st.session_state.selected_idx is not None:
            idx = st.session_state.selected_idx
            if idx < len(display_data):
                selected_item_data = display_data[idx]
                selected_item = display_data.pop(idx)
                display_data.insert(0, selected_item)
                is_reordered = True

        st.divider()
        st.subheader("Property Matches")
        
        # Show Selected Property card
        if is_reordered and selected_item_data:
            st.markdown("### 🎯 SELECTED PROPERTY")
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                c1.subheader(f"📍 {selected_item_data.get('Address')}")
                c1.caption(f"{selected_item_data.get('Suburb')} | {selected_item_data.get('CouncilArea')} | {selected_item_data.get('Type')}")
                c2.metric("Price", f"${selected_item_data.get('Price', 0):,.0f}")
                
                ic1, ic2, ic3, ic4 = st.columns(4)
                ic1.write(f"🛏 **{selected_item_data['Rooms']}** Beds")
                ic2.write(f"🚿 **{selected_item_data['Bathroom']}** Baths")
                ic3.write(f"🚗 **{selected_item_data['Car']}** Spots")
                ic4.write(f"📐 **{selected_item_data['BuildingArea']}** m²")
                
                st.markdown("---")
                if st.button("Generate Neighborhood Report", key="btn_selected", type="primary", use_container_width=True):
                    with st.spinner("Analyzing neighborhood..."):
                        try:
                            r = requests.get(f"{API_URL}/house_report", params={"lat": selected_item_data['Lattitude'], "lon": selected_item_data['Longtitude']})
                            if r.status_code == 200:
                                data_res = r.json()
                                if data_res.get("status") == "success": render_amenity_report(data_res.get("data"))
                        except Exception as e: st.error(f"Error: {e}")
            st.divider()

        st.subheader("All Properties")

        for i, house in enumerate(data):
            # Check if this card matches the one selected on the map
            is_this_selected = (st.session_state.selected_idx == i)
            card_highlight = "🎯 " if is_this_selected else ""
            
            with st.container(border=True):
                # 1. Header Row: Address and Price
                c1, c2 = st.columns([3, 1])
                c1.subheader(f"{card_highlight}{house.get('Address')}")
                c1.caption(f"{house.get('Suburb')} | {house.get('CouncilArea')} | {house.get('Type')}")
                c2.metric("Price", f"${house.get('Price', 0):,.0f}")
                
                # 2. Specs Row: Beds, Baths, etc.
                ic1, ic2, ic3, ic4 = st.columns(4)
                ic1.write(f"🛏 **{house['Rooms']}** Beds")
                ic2.write(f"🚿 **{house['Bathroom']}** Baths")
                ic3.write(f"🚗 **{house['Car']}** Spots")
                ic4.write(f"📐 **{house['BuildingArea']}** m²")
                
                # 3. Actions Row: Dedicated space for buttons away from the report area
                st.markdown("---") # Visual separator for actions
                btn_col1, btn_col2 = st.columns([1, 2])
                
                with btn_col1:
                    # Relocated "View on Map" button
                    if st.button("📍 View on Map", key=f"view_map_{i}", use_container_width=True):
                        st.session_state.selected_idx = i
                        st.rerun()
                
                with btn_col2:
                    # "More Details" remains clean. If amenities are generated, 
                    # they appear inside this expander without pushing the Map button away.
                    with st.expander("More Details & Neighborhood"):
                        if st.button("Scan Neighborhood", key=f"btn_list_{i}", use_container_width=True):
                            with st.spinner("Analyzing..."):
                                try:
                                    r = requests.get(f"{API_URL}/house_report", params={
                                        "lat": house['Lattitude'], 
                                        "lon": house['Longtitude']
                                    })
                                    if r.status_code == 200:
                                        data_res = r.json()
                                        if data_res.get("status") == "success": 
                                            # Amenities are rendered here, separate from the Map button
                                            render_amenity_report(data_res.get("data"))
                                except Exception as e: 
                                    st.error(f"Error: {e}")
                                    
# ==========================================
# 3. REAL ESTATE AGENT VIEW
# ==========================================
elif role == "Real Estate Agent" and st.session_state.authenticated:
    st.markdown("### 📋 Property Management Dashboard")
    tab_stats, tab_add, tab_manage = st.tabs(["📊 Analytics", "➕ Add Property", "🛠 Manage & Edit"])

    with tab_stats:
        st.caption("Market overview")
        c1, c2 = st.columns(2)
        region_stats = c1.selectbox("Region", ["Southern Metropolitan", "Northern Metropolitan", "Western Metropolitan"])
        type_stats = c2.selectbox("Type", ["h", "u", "t"])
        if st.button("Load Stats"):
            market_engine = stats.MarketStats()
            market_engine.show_dashboard(region_stats, type_stats)

    with tab_add:
        st.subheader("New Listing Entry")
        with st.form("add_house_form", clear_on_submit=False):
            st.markdown("#### 1. Location Details")
            c1, c2 = st.columns(2)
            new_sub = c1.text_input("Suburb", value=st.session_state.get("new_sub", "Richmond"), key="new_sub")
            new_addr = c2.text_input("Address", value=st.session_state.get("new_addr", "101 Church St"), key="new_addr")
            
            # Verify Address Button
            verify_col1, verify_col2 = st.columns([3, 1])
            with verify_col2:
                if st.form_submit_button("🔍 Verify", use_container_width=True):
                    # Read current inputs from session_state to avoid defaults on rerun
                    addr = st.session_state.get("new_addr", "").strip()
                    sub = st.session_state.get("new_sub", "").strip()
                    if addr and sub:
                        with st.spinner("Verifying address..."):
                            try:
                                r = requests.post(f"{API_URL}/verify_address", json={"address": addr, "suburb": sub})
                                if r.status_code == 200:
                                    result = r.json()
                                    st.session_state.verified_address = {
                                        "latitude": result["latitude"],
                                        "longitude": result["longitude"],
                                        "address": result["address"]
                                    }
                                    # Auto-fill lat/lon widgets
                                    st.session_state["new_lat"] = result["latitude"]
                                    st.session_state["new_lon"] = result["longitude"]
                                    st.session_state.address_verified = True
                                    st.success(f"✅ Address verified: {result['address']}")
                                else:
                                    st.error(f"❌ Address not found: {r.json().get('error', 'Unknown error')}")
                                    st.session_state.verified_address = None
                                    st.session_state.address_verified = False
                            except Exception as e:
                                st.error(f"❌ Error: {e}")
                                st.session_state.verified_address = None
                                st.session_state.address_verified = False
                    else:
                        st.error("Please enter both Address and Suburb")
            
            # We use an expander so they are hidden by default (Auto-detect),
            # but the agent can open it to override them manually.
            with st.expander("Advanced Location Details (Auto-detected)"):
                st.caption("Leave these as 'Auto-detect' to let the system calculate them based on the address.")
                
                ac1, ac2, ac3 = st.columns(3)
                # Add "Auto-detect" as the first option
                region_opts = ["Auto-detect", "Southern Metropolitan", "Northern Metropolitan", "Western Metropolitan", "Eastern Metropolitan", "South-Eastern Metropolitan", "Eastern Victoria", "Northern Victoria", "Western Victoria"]
                new_region = ac1.selectbox("Region", region_opts)
                
                council_opts = ["Auto-detect"] + COUNCILS
                new_council = ac2.selectbox("Council Area", council_opts)
                
                # Distance defaults to 0.0 (which API interprets as "calculate for me")
                new_dist = ac3.number_input("Dist from CBD (km)", 0.0, 100.0, 0.0, help="Leave 0.0 to auto-calculate")

            # --- Coordinates (Auto-filled by verification) ---
            # Use verified coordinates if available
            if st.session_state.verified_address:
                default_lat = st.session_state.verified_address["latitude"]
                default_lon = st.session_state.verified_address["longitude"]
                verified_status = "✅ Verified"
            else:
                default_lat = -37.8136
                default_lon = 144.9631
                verified_status = "⚠️ Not verified"
            
            st.caption(f"Address Status: {verified_status}")
            
            # Hidden coordinate inputs (or visible but read-only-ish)
            lc1, lc2 = st.columns(2)
            new_lat = lc1.number_input("Latitude", value=st.session_state.get("new_lat", default_lat), format="%.4f", key="new_lat")
            new_lon = lc2.number_input("Longitude", value=st.session_state.get("new_lon", default_lon), format="%.4f", key="new_lon")
            
            st.caption(f"Address Status: {verified_status}")
            if not st.session_state.address_verified:
                st.info("Please verify address successfully before publishing.")

            st.markdown("#### 2. Specs")
            s1, s2, s3, s4 = st.columns(4)
            new_rooms = s1.number_input("Rooms", 1, 10, 2)
            new_bath = s2.number_input("Bathrooms", 1, 5, 1)
            new_car = s3.number_input("Car Spots", 0, 5, 1)
            new_type = s4.selectbox("Type", ["h", "u", "t"], format_func=lambda x: {"h":"House", "u":"Unit", "t":"Townhouse"}[x])

            st.markdown("#### 3. Technical & Price")
            t1, t2, t3, t4 = st.columns(4)
            new_price = t1.number_input("Price ($)", 100000, 10000000, 850000, step=10000)
            new_land = t2.number_input("Landsize", 0, 10000, 400)
            new_build = t3.number_input("Building Area", 0, 5000, 120)
            new_year = t4.number_input("Year Built", 1800, 2025, 2010)
            new_prop_count = st.number_input("Property Count (Density)", 0, 25000, 4000)

            if st.form_submit_button("🚀 Publish Listing"):
                # Enforce address verification before allowing publish
                if not st.session_state.get("address_verified", False):
                    st.error("Cannot publish: address not verified or not found.")
                    st.stop()
                
                # [MODIFIED] Logic to send None if Auto-detect is selected
                final_region = new_region if new_region != "Auto-detect" else None
                final_council = new_council if new_council != "Auto-detect" else None
                final_dist = new_dist if new_dist > 0 else None # 0 means auto-calc

                payload = {
                    "Suburb": st.session_state.get("new_sub", new_sub), 
                    "Address": st.session_state.get("new_addr", new_addr), 
                    "Regionname": final_region, 
                    "CouncilArea": final_council,
                    "Lattitude": st.session_state.get("new_lat", new_lat), 
                    "Longtitude": st.session_state.get("new_lon", new_lon), 
                    "Distance": final_dist,
                    "Rooms": new_rooms, "Bathroom": new_bath, "Car": new_car, "Type": new_type,
                    "Price": new_price, "Landsize": new_land, "BuildingArea": new_build, "YearBuilt": new_year,
                    "Propertycount": new_prop_count, "SellerG": "Agent User"
                }
                try:
                    r = requests.post(f"{API_URL}/add_house", json=payload)
                    if r.status_code == 200:
                        addr_text = f"{st.session_state.get('new_addr', new_addr)}, {st.session_state.get('new_sub', new_sub)}"
                        st.success(f"Aggiunta casa in {addr_text}. ID: {r.json()['id']}")
                        st.cache_data.clear()
                        # Clear input state after successful publish
                        for k in ["new_sub", "new_addr", "new_lat", "new_lon", "verified_address"]:
                            if k in st.session_state:
                                del st.session_state[k]
                        st.session_state.address_verified = False
                    else: st.error(f"Error: {r.text}")
                except Exception as e: st.error(f"Error: {e}")

    with tab_manage:
        st.subheader("Search & Modify Inventory")
        search_query = st.text_input("🔍 Search by Address/Suburb", placeholder="Type address...", key="agent_search")
        if search_query:
            res = requests.get(f"{API_URL}/agent/search", params={"query": search_query})
            results = res.json().get("results", []) if res.status_code == 200 else []
            if not results: st.info("No matches found.")
            else:
                st.write(f"Found {len(results)} properties.")
                for house in results:
                    with st.container(border=True):
                        # Header with Address and Action Buttons
                        col_header, col_actions = st.columns([3, 1])
                        
                        with col_header:
                            st.markdown(f"### {house.get('Address', 'Address not available')}")
                            st.caption(f"📍 {house.get('Suburb')} | {house.get('CouncilArea', 'N/A')} | {house.get('Regionname', '')}")
                        
                        with col_actions:
                            # Buttons grouped to the right
                            b1, b2 = st.columns(2)
                            if b1.button("✏️ Edit", key=f"edit_{house['HouseID']}", help="Edit"):
                                st.session_state.edit_house_data = house
                                st.rerun()
                            if b2.button("🗑 Delete", key=f"del_{house['HouseID']}", type="primary", help="Delete"):
                                requests.delete(f"{API_URL}/remove_house", params={"id": house['HouseID']})
                                st.success("Removed!")
                                st.cache_data.clear()
                                st.rerun()

                        st.divider()

                        # Grid with all features
                        ic1, ic2, ic3, ic4 = st.columns(4)
                        
                        # Column 1: Price and Year
                        ic1.markdown(f"💰 **Price:** ${house.get('Price', 0):,.0f}")
                        ic1.markdown(f"📅 **Year:** {int(house.get('YearBuilt', 0))}")
                        
                        # Column 2: Rooms and Bathrooms
                        ic2.markdown(f"🛏 **Beds:** {house.get('Rooms')}")
                        ic2.markdown(f"🚿 **Baths:** {house.get('Bathroom')}")
                        
                        # Column 3: Car and Type
                        type_map = {"h": "House", "u": "Unit", "t": "Townhouse"}
                        house_type = type_map.get(house.get('Type'), house.get('Type'))
                        ic3.markdown(f"🚗 **Car:** {house.get('Car')}")
                        ic3.markdown(f"🏠 **Type:** {house_type}")
                        
                        # Column 4: Dimensions
                        ic4.markdown(f"🌍 **Land:** {house.get('Landsize')} m²")
                        ic4.markdown(f"🏗 **Build:** {house.get('BuildingArea')} m²")
                        
                        # Expandable extra details
                        with st.expander("Technical Details (Distance, Lat/Lon)"):
                            st.write(f"**CBD Distance:** {house.get('Distance')} km")
                            st.write(f"**Latitude:** {house.get('Lattitude')}")
                            st.write(f"**Longitude:** {house.get('Longtitude')}")
                            st.write(f"**Density (Property Count):** {house.get('Propertycount')}")

        st.divider()

        if st.session_state.edit_house_data:
            target = st.session_state.edit_house_data
            st.markdown(f"### ✏️ Editing: {target['Address']}")
            st.info("Modify the technical values below and click Save.")

            with st.form("edit_house_form"):
                ec1, ec2, ec3 = st.columns(3)
                e_price = ec1.number_input("Price", value=safe_val(target.get('Price'), float, 0.0))
                e_rooms = ec2.number_input("Rooms", value=safe_val(target.get('Rooms'), int, 1))
                e_bath = ec3.number_input("Bathrooms", value=safe_val(target.get('Bathroom'), int, 1))
                
                ec4, ec5, ec6 = st.columns(3)
                e_car = ec4.number_input("Car Spots", value=safe_val(target.get('Car'), int, 0))
                e_land = ec5.number_input("Landsize", value=safe_val(target.get('Landsize'), float, 0.0))
                e_build = ec6.number_input("Building Area", value=safe_val(target.get('BuildingArea'), float, 0.0))
                
                ec7, ec8, ec9 = st.columns(3)
                e_year = ec7.number_input("Year Built", value=safe_val(target.get('YearBuilt'), float, 2000.0))
                
                # Type Indexing
                curr_type = target.get('Type', 'h')
                e_type = ec8.selectbox("Type", ["h", "u", "t"], index=["h", "u", "t"].index(curr_type) if curr_type in ["h","u","t"] else 0)
                
                # [NEW] Council Edit
                curr_council = target.get('CouncilArea', COUNCILS[0])
                if curr_council not in COUNCILS: COUNCILS.append(curr_council) # handle unknown
                e_council = ec9.selectbox("Council Area", COUNCILS, index=COUNCILS.index(curr_council))

                # Hidden
                e_dist = safe_val(target.get('Distance'), float, 0.0)
                e_prop = safe_val(target.get('Propertycount'), float, 0.0)
                e_lat = safe_val(target.get('Lattitude'), float, -37.81)
                e_lon = safe_val(target.get('Longtitude'), float, 144.96)
                e_reg = target.get('Regionname', "Southern Metropolitan")
                e_sub = target.get('Suburb', "Unknown")
                e_addr = target.get('Address', "Unknown")

                if st.form_submit_button("💾 Save Changes"):
                    update_payload = {
                        "HouseID": target['HouseID'],
                        "Price": e_price, "Rooms": e_rooms, "Bathroom": e_bath,
                        "Car": e_car, "Landsize": e_land, "BuildingArea": e_build,
                        "YearBuilt": e_year, "Type": e_type, "CouncilArea": e_council,
                        "Distance": e_dist, "Propertycount": e_prop,
                        "Lattitude": e_lat, "Longtitude": e_lon,
                        "Regionname": e_reg, "Suburb": e_sub, "Address": e_addr,
                        "SellerG": target.get("SellerG", "Agent")
                    }
                    try:
                        ur = requests.put(f"{API_URL}/update_house", json=update_payload)
                        if ur.status_code == 200:
                            st.success("Property updated successfully!")
                            st.session_state.edit_house_data = None
                            st.cache_data.clear()
                            st.rerun()
                        else: st.error(f"Update failed: {ur.text}")
                    except Exception as e: st.error(f"Error: {e}")
            
            if st.button("Cancel Edit"):
                st.session_state.edit_house_data = None
                st.rerun()