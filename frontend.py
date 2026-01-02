import streamlit as st
import requests
import pandas as pd
import os

# Use Docker network URL if available, otherwise fallback to localhost
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Melb House Finder", layout="wide")
st.title("Melbourne Housing Recommender")

# --- Initialize Session State ---
# This keeps the results in memory even when you click other buttons
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "search_performed" not in st.session_state:
    st.session_state.search_performed = False

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Parameters")
    
    # Numerical
    rooms = st.slider("Rooms", 1, 10, 3)
    price = st.number_input("Target Price ($)", 100000, 10000000, 1000000, step=50000)
    dist = st.slider("Max Distance from CBD (km)", 1.0, 50.0, 10.0)
    bathroom = st.slider("Bathrooms", 1, 5, 2)
    car = st.slider("Car Spots", 0, 5, 1)
    land = st.number_input("Landsize (sqm)", 0, 10000, 600)
    build_area = st.number_input("Building Area (sqm)", 0, 5000, 150)
    prop_count = st.slider("Suburb Density (Property Count)", 0, 22000, 5000, 
                           help="Higher number = more dense suburb")
    year_built = st.number_input("Year Built (Approx)", 1800, 2024, 2000)

    # Categorical
    type_options = {
        "h": "House/Cottage/Villa",
        "u": "Unit/Duplex",
        "t": "Townhouse",
        "br": "Bedroom(s)",
        "dev site": "Development Site",
        "o res": "Other Residential"
    }
    type_code = st.selectbox("Property Type", options=list(type_options.keys()), 
                             format_func=lambda x: type_options[x])

    region = st.selectbox("Region", [
        "Southern Metropolitan", "Northern Metropolitan", "Western Metropolitan", 
        "Eastern Metropolitan", "South-Eastern Metropolitan", "Eastern Victoria",
        "Northern Victoria", "Western Victoria"
    ])

    # When this is clicked, we will update the session state
    search = st.button("Find Matching Houses")

# --- Logic to Fetch Data ---
if search:
    payload = {
        "Rooms": rooms, "Price": price, "Distance": dist, "Bathroom": bathroom,
        "Car": car, "Landsize": land, "BuildingArea": build_area,
        "Propertycount": prop_count, "YearBuilt": year_built,
        "Type": type_code, "Regionname": region
    }
    
    with st.spinner("Calculating similarity..."):
        try:
            res = requests.post(f"{API_URL}/recommend", json=payload)
            if res.status_code == 200:
                # Store results in session state
                st.session_state.recommendations = res.json().get("recommendations", [])
                st.session_state.search_performed = True
            else:
                st.error(f"Error {res.status_code}: {res.text}")
        except Exception as e:
            st.error(f"Connection Error: {e}")

# --- Main Page Display ---
# We check session_state instead of the button directly
if st.session_state.search_performed:
    data = st.session_state.recommendations
    
    if not data:
        st.warning("No matches found.")
    else:
        st.success("Top 5 Recommendations based on your criteria:")
        
        # Create a map dataframe
        map_data = pd.DataFrame(data)[['Lattitude', 'Longtitude']].dropna()
        map_data.columns = ['lat', 'lon']
        st.map(map_data)

        for i, house in enumerate(data):
            with st.container():
                st.markdown(f"### {i+1}. {house.get('Address', 'Unknown Address')}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Price", f"${house['Price']:,.0f}")
                c2.metric("Suburb", house['Suburb'])
                c3.metric("Seller", house['SellerG'])
                
                st.markdown(f"""
                **Details:** {house['Rooms']} Beds | {house['Bathroom']} Baths | {house['Car']} Cars  
                **Type:** {type_options.get(house['Type'], house['Type'])}  
                **Method:** {house['Method']} (Sale Type)
                """)
                
                # --- Geospatial Check Button ---
                # This button is now stable because 'data' persists in session_state
                if st.button(f"Generate Location Report for House #{i+1}", key=f"geo_{i}"):
                    with st.spinner("Analyzing neighborhood infrastructure..."):
                        
                        try:
                            report_res = requests.get(f"{API_URL}/house_report", 
                                                    params={"lat": house['Lattitude'], "lon": house['Longtitude']})
                            report_data = report_res.json()

                            if report_data["status"] == "success":
                                rdata = report_data["data"]
                                
                                # Display results in columns (Updated with new fields)
                                rc1, rc2, rc3 = st.columns(3)
                                
                                with rc1:
                                    st.markdown("**Transport**")
                                    st.write(f"Train: {rdata.get('train_station') or 'N/A'}m")
                                    st.write(f"Tram: {rdata.get('tram_stop') or 'N/A'}m")
                                    st.write(f"Bus: {rdata.get('bus_stop') or 'N/A'}m")
                                
                                with rc2:
                                    st.markdown("**Education & Services**")
                                    st.write(f"School: {rdata.get('school') or 'N/A'}m")
                                    st.write(f"Hospital: {rdata.get('hospital') or 'N/A'}m")
                                    st.write(f"Supermarket: {rdata.get('supermarket') or 'N/A'}m")
                                
                                with rc3:
                                    st.markdown("**Nature & Lifestyle**")
                                    st.write(f"Park: {rdata.get('park') or 'N/A'}m")
                                    st.write(f"Lake: {rdata.get('lake') or 'N/A'}m")
                                    st.write(f"Cafe: {rdata.get('cafe') or 'N/A'}m")
                                    st.write(f"Gym: {rdata.get('gym') or 'N/A'}m")
                                    
                            else:
                                st.error("Could not fetch location data.")
                        except Exception as e:
                            st.error(f"Error fetching report: {e}")
                
                st.divider()