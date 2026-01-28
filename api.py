from flask import Flask, request, jsonify, g
import pandas as pd
import numpy as np
import uuid
import os
import json
from typing import Optional
from pydantic import BaseModel, Field, ValidationError
from geo import GeoService
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from geopy.distance import geodesic
from model_manager import ModelManager
from train import train
from drift import compare_latest_models

import logging
from logging.handlers import RotatingFileHandler
import time

app = Flask(__name__)

# --- Configuration ---
geo_service = GeoService()
geocoder = Nominatim(user_agent="melbourne_housing_app")
manager = ModelManager()
CBD_COORDS = (-37.8136, 144.9631)

# --- Buffer configuration ---
RETRAIN_THRESHOLD = 5  # Number of new houses to trigger retraining
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PENDING_DATA_FILE = os.path.join(BASE_DIR, 'pending_houses.jsonl')
pending_count = 0

# Global State
model_pipeline = None
database = None
preprocessor = None
features_list = []
last_drift_check = None  # Stores result of last model comparison

# --- Logging Setup ---
logger = logging.getLogger('melbourne_housing_api')
logger.setLevel(logging.INFO)

# Keep logs in a dedicated file, max 10MB each, keep last 5
handler = RotatingFileHandler('app_metrics.log', maxBytes=10*1024*1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)



@app.before_request
def start_timer():
    g.start_time = time.time() 

@app.after_request
def log_request(response):
    if request.path != '/health':
        latency = time.time() - getattr(g, 'start_time', time.time())
        logger.info(f"Method: {request.method} | Path: {request.path} | "
                    f"Status: {response.status_code} | Latency: {latency:.4f}s")
    return response


# --- Data Models ---
class UserPreferences(BaseModel):
    # Numerical features
    Rooms: Optional[int] = Field(None, description="Number of rooms")
    Price: Optional[float] = Field(None, description="Max Price")
    Distance: Optional[float] = Field(None, description="Distance from CBD")
    Bedroom2: Optional[int] = Field(None, description="Scraped Bedrooms")
    Bathroom: Optional[int] = Field(None, description="Number of bathrooms")
    Car: Optional[int] = Field(None, description="Car spots")
    Landsize: Optional[float] = Field(None, description="Land size in sqm")
    BuildingArea: Optional[float] = Field(None, description="Building size in sqm")
    YearBuilt: Optional[float] = Field(None, description="Year of construction")
    Propertycount: Optional[float] = Field(None, description="Number of properties in suburb")
    Lattitude: Optional[float] = Field(None, description="Latitude")
    Longtitude: Optional[float] = Field(None, description="Longitude")
    
    # Categorical features
    Type: str = Field("h", description="Type: h=house, u=unit, t=townhouse")
    Regionname: Optional[str] = Field(None, description="Region Name")
    CouncilArea: Optional[str] = Field(None, description="Council Area") 
    Method: str = Field("S", description="Sale Method: S, SP, PI, VB, SA")

class NewHouse(UserPreferences):
    Suburb: str = Field(..., description="Suburb Name")
    Address: str = Field(..., description="Street Address")
    SellerG: str = Field("Private", description="Seller Agency")

class UpdateHouse(NewHouse):
    HouseID: str = Field(..., description="Unique ID is required for updates")

# --- Core Logic ---

def load_system():
    global model_pipeline, database, preprocessor, features_list
    
    # Attempt 1: Load existing model and data
    try:
        print("Loading system artifacts...")
        artifacts = manager.load_current_model()
    except Exception as e:
        # If load of the model fails, start Cold Start
        print(f"Model not found or corrupted ({e}). Initiating COLD START training...")
        
        try:
            # Perform "on-the-fly" training
            train()
            print("Cold start training complete. Reloading system...")
            
            # Attempt 2: Retry loading after training
            artifacts = manager.load_current_model()
            
        except Exception as e2:
            print(f"CRITICAL ERROR: Auto-training failed. Check if 'melb_data.csv' is present.")
            print(f"Details: {e2}")
            return # Interrupt startup if training also fails

    try:
        model_pipeline = artifacts["pipeline"]
        
        # Handle different naming conventions
        database = artifacts.get("reference_data")
        if database is None:
            database = artifacts.get("database")
            
        if database is None:
            raise KeyError("Neither 'reference_data' nor 'database' found in model artifacts.")

        features_list = artifacts.get("features", [])
        if not features_list:
             features_list = artifacts.get("numeric_features", []) + artifacts.get("categorical_features", [])

        preprocessor = model_pipeline.named_steps['preprocessor']
        
        # Ensure IDs exist
        if 'HouseID' not in database.columns:
            print("Adding missing HouseIDs to loaded data")
            database['HouseID'] = [str(uuid.uuid4()) for _ in range(len(database))]
            
        print(f"System Loaded Successfully. {len(database)} houses in database.")
        
    except Exception as e:
        print(f"Error extracting artifacts: {e}")


def save_system(note="Auto-update"):
    """Saves the current state of the database and model."""
    if model_pipeline is None or database is None:
        print("Cannot save system: State is not initialized.")
        return

    artifacts = {
        "pipeline": model_pipeline,
        "reference_data": database,
        "features": features_list
    }
    # Save via Manager
    manager.save_model(artifacts, note=note)

def geo_fill(data: dict) -> dict:
    """
    Auto-detects Region, Council, and Distance based on Suburb and Coordinates.
    """
    if database is None: return data

    # 1. Distance Calculation (Lat/Lon -> CBD)
    # We can do this because strict verification in add_house ensures we have coordinates
    if data.get('Distance') is None and data.get('Lattitude') is not None and data.get('Longtitude') is not None:
        try:
            house_coords = (data['Lattitude'], data['Longtitude'])
            dist = geodesic(house_coords, CBD_COORDS).km
            data['Distance'] = round(dist, 1)
            print(f"Geo Fill: Calculated Distance -> {dist:.1f} km")
        except Exception as e: 
            print(f"Geo Fill Distance error: {e}")

    # 2. Region & Council Inference (from Suburb neighbors)
    suburb = data.get('Suburb')
    if suburb:
        neighbors = database[database['Suburb'].str.lower() == suburb.lower()]
        
        if not neighbors.empty:
            # If Regionname is missing, take the most frequent in that suburb
            if not data.get('Regionname') or data.get('Regionname') == "Auto-detect":
                mode_reg = neighbors['Regionname'].mode()
                if not mode_reg.empty: 
                    data['Regionname'] = mode_reg[0]
                    print(f"Geo Fill: Region inferred -> {data['Regionname']}")
            
            # If CouncilArea is missing, take the most frequent
            if not data.get('CouncilArea') or data.get('CouncilArea') == "Auto-detect":
                mode_counc = neighbors['CouncilArea'].mode()
                if not mode_counc.empty: 
                    data['CouncilArea'] = mode_counc[0]
                    print(f"Geo Fill: Council inferred -> {data['CouncilArea']}")
                
            # If Propertycount is missing
            if data.get('Propertycount') is None:
                data['Propertycount'] = neighbors['Propertycount'].median()

    # Fallbacks to prevent pipeline crashes if suburb is totally new
    if not data.get('Regionname'): data['Regionname'] = "Southern Metropolitan"
    if not data.get('CouncilArea'): data['CouncilArea'] = "Unknown"
    
    return data


def smart_fill(input_data: dict) -> dict:
    """
    Implements Smart Lookup.
    Fills missing values (None or 0) using the median of similar houses 
    (same Suburb and Type) from the loaded database.
    """
    if database is None: return input_data # Safety check
    
    data = input_data.copy()
    
    data = geo_fill(data)

    # 1. Identify context (Suburb & Type)
    suburb = data.get('Suburb')
    h_type = data.get('Type', 'h')
    
    # 2. Filter Database for "Similar Houses"
    mask = (database['Suburb'] == suburb) & (database['Type'] == h_type)
    similar_houses = database[mask]
    
    # Fallback: if no houses in that suburb+type, try just suburb
    if similar_houses.empty and suburb:
        similar_houses = database[database['Suburb'] == suburb]
        
    # Fallback 2: if still empty (new suburb), use whole DB
    if similar_houses.empty:
        similar_houses = database

    # 3. Fields to impute
    fields_config = {
        'Price': {'allow_zero': False},
        'Rooms': {'allow_zero': False},
        'Bedroom2': {'allow_zero': False},
        'Bathroom': {'allow_zero': False},
        'Car': {'allow_zero': True}, # 0 cars is valid
        'Landsize': {'allow_zero': False}, 
        'BuildingArea': {'allow_zero': False},
        'YearBuilt': {'allow_zero': False},
        'Distance': {'allow_zero': True},
        'Propertycount': {'allow_zero': False}
    }

    print(f"Smart Fill running for {suburb} ({h_type}). Found {len(similar_houses)} neighbors.")

    for field, config in fields_config.items():
        val = data.get(field)
        
        # Check if value is "missing" (None or 0 where 0 is invalid)
        is_missing = val is None or (val == 0 and not config['allow_zero'])
        
        if is_missing:
            # Calculate median from neighbors
            median_val = similar_houses[field].median()
            
            # If neighbor median is NaN, use global median from whole DB
            if pd.isna(median_val):
                median_val = database[field].median()
            
            # Apply value
            if pd.notna(median_val):
                data[field] = float(median_val) if 'float' in str(type(median_val)) else int(median_val)
                # Ensure Bedroom2 matches Rooms if still missing
                if field == 'Bedroom2' and data.get('Rooms'):
                     data['Bedroom2'] = data['Rooms']

    return data

def retrain_model(note="Online Learning"):
    """Refits the KNN model on the current database (Online Learning)."""
    if model_pipeline is None: return

    print("Retraining KNN model on updated data...")
    try:
        # Transform current database using the pre-trained preprocessor
        X_transformed = preprocessor.transform(database)
        
        # Refit only the KNN part
        knn = model_pipeline.named_steps['model']
        knn.fit(X_transformed)
        
        # Update pipeline step
        model_pipeline.steps[-1] = ('model', knn)
        
        save_system(note=note)
        print("Retraining complete and saved.")
    except Exception as e:
        print(f"Error during retraining: {e}")



# --- Utility per il Buffer ---

def append_to_buffer(house_data):
    try:
        with open(PENDING_DATA_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(house_data) + '\n')
            f.flush()
        print(f"--- FILE SAVED IN: {os.path.abspath(PENDING_DATA_FILE)} ---")
    except Exception as e:
        print(f"Errore: {e}")

def clear_buffer():
    """Clears the buffer file after retraining."""
    if os.path.exists(PENDING_DATA_FILE):
        os.remove(PENDING_DATA_FILE)

def load_pending_on_startup():
    global database, pending_count
    if os.path.exists(PENDING_DATA_FILE):
        try:
            with open(PENDING_DATA_FILE, 'r') as f:
                pending_list = [json.loads(line) for line in f if line.strip()]
            
            if pending_list:
                df_pending = pd.DataFrame(pending_list)
                if database is not None:
                    database = pd.concat([database, df_pending], ignore_index=True)
                else:
                    database = df_pending
                
                pending_count = len(pending_list)
                print(f"Loaded {pending_count} pending records from file.")
                
                # If on startup we already exceed the threshold, immediate retrain
                if pending_count >= RETRAIN_THRESHOLD:
                    retrain_model("Startup batch retrain")
                    clear_buffer()
                    pending_count = 0
        except Exception as e:
            print(f"Error loading startup buffer: {e}")



# --- Routes ---

@app.route('/health', methods=['GET'])
def health():
    status = "online" if database is not None and model_pipeline is not None else "uninitialized"
    count = len(database) if database is not None else 0
    return jsonify({"status": status, "houses_count": count})

@app.route('/recommend', methods=['POST'])
def recommend():
    if model_pipeline is None or preprocessor is None:
        return jsonify({"error": "System not initialized. Check server logs."}), 503
    
    try:
        data = request.json
        _ = data.pop('required_amenities', None)
        
        # 1. Parse Input
        prefs = UserPreferences(**data)
        input_dict = prefs.dict()
        
        # 2. Apply Smart Fill
        input_dict = smart_fill(input_dict)
        
        # 3. Create DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # 4. Transform & Predict
        processed_input = preprocessor.transform(input_df)
        knn = model_pipeline.named_steps['model']
        distances, indices = knn.kneighbors(processed_input)
        
        results = []
        for idx in indices[0]:
            if idx < len(database):
                house = database.iloc[idx].to_dict()
                house = {k: (v if pd.notna(v) else None) for k, v in house.items()}
                results.append(house)
        
        return jsonify({"recommendations": results})
    except ValidationError as ve:
        return jsonify({"error": "Validation Error", "details": ve.errors()}), 400
    except Exception as e:
        print(f"Error in recommend: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/house_report', methods=['GET'])
def house_report():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    if lat is None or lon is None: return jsonify({"error": "Missing params"}), 400
    return jsonify(geo_service.scan_area(lat, lon))

@app.route('/verify_address', methods=['POST'])
def verify_address():
    try:
        data = request.json
        address = data.get('address', '').strip()
        suburb = data.get('suburb', '').strip()
        if not address: return jsonify({"error": "Address required"}), 400
        full_address = f"{address}, {suburb}, Victoria, Australia" if suburb else f"{address}, Victoria, Australia"
        location = geocoder.geocode(full_address, timeout=5)
        if location is None: return jsonify({"error": "Address not found"}), 404
        return jsonify({
            "status": "success", "address": location.address,
            "latitude": location.latitude, "longitude": location.longitude
        })
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/agent/search', methods=['GET'])
def search_houses():
    if database is None: return jsonify({"results": []})
    query = request.args.get('query', '').lower()
    if len(query) < 2: return jsonify({"results": []})
    
    mask = (database['Address'].astype(str).str.lower().str.contains(query, na=False) | 
            database['Suburb'].astype(str).str.lower().str.contains(query, na=False))
            
    matches = database[mask].head(20).to_dict(orient='records')
    clean_matches = [{k: (v if pd.notna(v) else None) for k, v in h.items()} for h in matches]
    return jsonify({"results": clean_matches})


@app.route('/add_house', methods=['POST'])
def add_house():
    global database, pending_count
    if database is None: return jsonify({"error": "System not initialized"}), 503
    
    try:
        data = request.json
        
        # --- GEO-VERIFICATION ---
        if data.get('Lattitude') is None or data.get('Longtitude') is None:
            address = data.get('Address', '').strip()
            suburb = data.get('Suburb', '').strip()
            
            if not address or not suburb:
                 return jsonify({"error": "Address and Suburb are required for verification."}), 400

            print(f"Attempting strict verification for: {address}, {suburb}")
            full_address = f"{address}, {suburb}, Victoria, Australia"
            
            try:
                location = geocoder.geocode(full_address, timeout=5)
                
                # CONTROL BLOCK: if the right address is not available, block the insertion
                if location is None:
                    return jsonify({
                        "error": "Strict Verification Failed: Address not found on maps. House NOT added."
                    }), 400
                
                # Verificated coordinates injection
                data['Lattitude'] = location.latitude
                data['Longtitude'] = location.longitude
                print(f"Verified: {location.address} ({location.latitude}, {location.longitude})")
                
            except Exception as e:
                return jsonify({"error": f"Geocoding service unavailable: {str(e)}"}), 503
        
        # 1. Validation Smart Fill
        new_house_model = NewHouse(**data)
        new_house_dict = new_house_model.dict()
        
        # Apply Smart Fill (Contextual Defaults per Price, Rooms, ecc.)
        final_house_data = smart_fill(new_house_dict)
        final_house_data['HouseID'] = str(uuid.uuid4())
        
        # Align DataFrame columns
        for col in database.columns:
            if col not in final_house_data:
                final_house_data[col] = None
                
        # 2. Saving record in Database In-Memory and Buffer JSONL
        database = pd.concat([database, pd.DataFrame([final_house_data])], ignore_index=True)
        append_to_buffer(final_house_data)
        pending_count += 1
        
        # 3. Retrain at a given treshold
        status_msg = f"House added to buffer ({pending_count}/{RETRAIN_THRESHOLD})"
        drift_result = None
        if pending_count >= RETRAIN_THRESHOLD:
            print(f"Treshold achived ({pending_count}). Start retraining batch...")
            retrain_model(note=f"Batch retraining after {pending_count} inserts")
            
            # Automatic drift check after retrain
            global last_drift_check
            try:
                drift_result = compare_latest_models()
                last_drift_check = drift_result
                print(f"Drift check completed. Full dataset drift: {drift_result.get('drift', {}).get('should_retrain', False)}")
            except Exception as e:
                print(f"Drift check failed: {e}")
                last_drift_check = {"error": str(e)}
            
            clear_buffer()
            pending_count = 0
            status_msg = "Threshold reached: Model retrained and buffer cleared."
        
        return jsonify({
            "message": "House successfully verified and added", 
            "id": final_house_data['HouseID'],
            "status": status_msg,
            "coords": {"lat": final_house_data['Lattitude'], "lon": final_house_data['Longtitude']},
            "drift_check": drift_result
        })
        
    except ValidationError as e: return jsonify({"error": e.errors()}), 400
    except Exception as e: return jsonify({"error": str(e)}), 500


@app.route('/update_house', methods=['PUT'])
def update_house():
    if database is None: return jsonify({"error": "System not initialized"}), 503
    try:
        data = request.json
        update_data = UpdateHouse(**data).dict()
        house_id = update_data.pop('HouseID')
        
        if 'HouseID' not in database.columns:
             return jsonify({"error": "Database corrupt: No HouseID column"}), 500

        if house_id not in database['HouseID'].values: 
            return jsonify({"error": "ID not found"}), 404
            
        idx = database.index[database['HouseID'] == house_id].tolist()[0]
        for key, value in update_data.items():
            if key in database.columns: 
                database.at[idx, key] = value
                
        retrain_model(note=f"Updated house {house_id}")
        return jsonify({"message": "House updated"})
    except ValidationError as e: return jsonify({"error": e.errors()}), 400

@app.route('/remove_house', methods=['DELETE'])
def remove_house():
    global database
    if database is None: return jsonify({"error": "System not initialized"}), 503
    
    house_id = request.args.get('id')
    if not house_id or house_id not in database['HouseID'].values: 
        return jsonify({"error": "ID not found"}), 404
        
    database = database[database['HouseID'] != house_id].reset_index(drop=True)
    retrain_model(note=f"Removed house {house_id}")
    return jsonify({"message": "House removed"})

@app.route('/buffer_status', methods=['GET'])
def buffer_status():
    """Return current state of the buffer."""
    content = []
    if os.path.exists(PENDING_DATA_FILE):
        with open(PENDING_DATA_FILE, 'r') as f:
            content = [json.loads(line) for line in f if line.strip()]
    
    return jsonify({
        "pending_count_variable": pending_count,
        "file_exists": os.path.exists(PENDING_DATA_FILE),
        "records_in_file": len(content),
        "threshold": RETRAIN_THRESHOLD,
        "remaining_until_retrain": max(0, RETRAIN_THRESHOLD - pending_count),
        "buffer_content": content
    })

@app.route('/drift_status', methods=['GET'])
def drift_status():
    """Returns the last drift check result after retrain."""
    if last_drift_check is None:
        return jsonify({"status": "no_retrain_yet", "message": "No retrain has occurred yet."})
    return jsonify(last_drift_check)

if __name__ == '__main__':
    load_system() 
    load_pending_on_startup() 
    
    print("\n--- Registered routes ---")
    for rule in app.url_map.iter_rules():
        print(f"{rule.endpoint}: {rule.rule}")
    print("------------------------\n")
    
    app.run(host='0.0.0.0', port=8000, debug=False)