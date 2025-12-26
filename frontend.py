import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Melb House Finder", layout="wide")
st.title("🏡 Melbourne Housing Recommender")

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

    search = st.button("Find Matching Houses")

# --- Main Page ---
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
                data = res.json().get("recommendations", [])
                
                if not data:
                    st.warning("No matches found.")
                else:
                    # Visualizing results
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
                            c3.metric("Seller", house['SellerG']) # Displaying Seller even if not used in logic
                            
                            st.markdown(f"""
                            **Details:** {house['Rooms']} Beds | {house['Bathroom']} Baths | {house['Car']} Cars  
                            **Type:** {type_options.get(house['Type'], house['Type'])}  
                            **Method:** {house['Method']} (Sale Type)
                            """)
                            
                            # Geospatial Check Button
                            if st.button(f"🔍 Check Amenities for House #{i+1}", key=f"geo_{i}"):
                                with st.spinner("Querying Satellite/OSM Data..."):
                                    # Check for Lake
                                    geo_res = requests.get(f"{API_URL}/amenities", 
                                        params={"lat": house['Lattitude'], "lon": house['Longtitude'], "amenity": "lake"})
                                    geo_data = geo_res.json()
                                    
                                    if geo_data.get('found'):
                                        st.success(f"✅ {geo_data['message']}")
                                    else:
                                        st.info("❌ No lake immediately nearby.")
                            
                            st.divider()
            else:
                st.error(f"Error {res.status_code}: {res.text}")
        except Exception as e:
            st.error(f"Connection Error: {e}")