from flask import Flask, request, jsonify
import pandas as pd
import uuid
import os
from pydantic import BaseModel, Field, ValidationError
from geo import GeoService
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from model_manager import ModelManager # <--- Import

app = Flask(__name__)

# --- Configuration ---
geo_service = GeoService()
geocoder = Nominatim(user_agent="melbourne_housing_app")
manager = ModelManager() # Initialize Manager

# Global State
model_pipeline = None
database = None
preprocessor = None
numeric_features = []

# --- Data Models (Same as before) ---
class UserPreferences(BaseModel):
    Rooms: int = Field(3)
    Price: float = Field(1000000)
    Distance: float = Field(10.0)
    Bathroom: int = Field(2)
    Car: int = Field(1)
    Landsize: float = Field(500.0)
    BuildingArea: float = Field(150.0)
    Propertycount: float = Field(4000.0)
    YearBuilt: float = Field(2000.0)
    Type: str = Field("h")
    Regionname: str = Field("Southern Metropolitan")
    CouncilArea: str = Field("Yarra City Council")

class NewHouse(UserPreferences):
    Suburb: str = Field(..., description="Suburb Name")
    Address: str = Field(..., description="Street Address")
    SellerG: str = Field("Private", description="Seller Agency")
    Lattitude: float = Field(..., description="Geo Lat")
    Longtitude: float = Field(..., description="Geo Lon")

class UpdateHouse(NewHouse):
    HouseID: str = Field(..., description="Unique ID is required for updates")

# --- Core Logic ---

def load_system():
    global model_pipeline, database, preprocessor, numeric_features
    try:
        # Load via Manager
        artifacts = manager.load_current_model()
        
        model_pipeline = artifacts["pipeline"]
        database = artifacts["database"]
        numeric_features = artifacts["numeric_features"]
        preprocessor = model_pipeline.named_steps['preprocessor']
        
        # Ensure IDs exist
        if 'HouseID' not in database.columns:
            database['HouseID'] = [str(uuid.uuid4()) for _ in range(len(database))]
            save_system(note="Added missing HouseIDs")
            
        print(f"System Loaded. {len(database)} houses in database.")
    except Exception as e:
        print(f"⚠️ Critical Error loading system: {e}")
        print("Ensure train.py has been run at least once.")

def save_system(note="Auto-update"):
    artifacts = {
        "pipeline": model_pipeline,
        "database": database,
        "numeric_features": numeric_features,
        "categorical_features": getattr(model_pipeline, "categorical_features", ['Type', 'Regionname', 'CouncilArea']) 
    }
    # Save via Manager (creates new version)
    manager.save_model(artifacts, note=note)

def retrain_model(note="Online Learning"):
    """Online Learning: Refits the KNN model on the current database."""
    print("Retraining KNN model...")
    X_transformed = preprocessor.transform(database)
    knn = model_pipeline.named_steps['knn']
    knn.fit(X_transformed)
    model_pipeline.steps[-1] = ('knn', knn)
    save_system(note=note)

# --- Routes (Logic remains mostly the same, just imports changed) ---

@app.route('/health', methods=['GET'])
def health():
    status = "online" if database is not None else "uninitialized"
    count = len(database) if database is not None else 0
    return jsonify({"status": status, "houses_count": count})

@app.route('/recommend', methods=['POST'])
def recommend():
    if model_pipeline is None: return jsonify({"error": "Model not loaded"}), 503
    try:
        data = request.json
        _ = data.pop('required_amenities', [])
        prefs = UserPreferences(**data)
        input_df = pd.DataFrame([prefs.dict()])
        processed_input = preprocessor.transform(input_df)
        distances, indices = model_pipeline.named_steps['knn'].kneighbors(processed_input)
        
        results = []
        for idx in indices[0]:
            house = database.iloc[idx].to_dict()
            house = {k: (v if pd.notna(v) else None) for k, v in house.items()}
            results.append(house)
        return jsonify({"recommendations": results})
    except Exception as e:
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
    mask = (database['Address'].str.lower().str.contains(query, na=False) | 
            database['Suburb'].str.lower().str.contains(query, na=False))
    matches = database[mask].head(20).to_dict(orient='records')
    clean_matches = [{k: (v if pd.notna(v) else None) for k, v in h.items()} for h in matches]
    return jsonify({"results": clean_matches})

@app.route('/add_house', methods=['POST'])
def add_house():
    global database
    try:
        data = request.json
        new_house_data = NewHouse(**data).dict()
        new_house_data['HouseID'] = str(uuid.uuid4())
        database = pd.concat([database, pd.DataFrame([new_house_data])], ignore_index=True)
        retrain_model(note=f"Added house {new_house_data['Address']}")
        return jsonify({"message": "House added", "id": new_house_data['HouseID']})
    except ValidationError as e: return jsonify({"error": e.errors()}), 400

@app.route('/update_house', methods=['PUT'])
def update_house():
    try:
        data = request.json
        update_data = UpdateHouse(**data).dict()
        house_id = update_data.pop('HouseID')
        if house_id not in database['HouseID'].values: return jsonify({"error": "ID not found"}), 404
        idx = database.index[database['HouseID'] == house_id].tolist()[0]
        for key, value in update_data.items():
            if key in database.columns: database.at[idx, key] = value
        retrain_model(note=f"Updated house {house_id}")
        return jsonify({"message": "House updated"})
    except ValidationError as e: return jsonify({"error": e.errors()}), 400

@app.route('/remove_house', methods=['DELETE'])
def remove_house():
    global database
    house_id = request.args.get('id')
    if not house_id or house_id not in database['HouseID'].values: return jsonify({"error": "ID not found"}), 404
    database = database[database['HouseID'] != house_id].reset_index(drop=True)
    retrain_model(note=f"Removed house {house_id}")
    return jsonify({"message": "House removed"})

if __name__ == '__main__':
    load_system()
    app.run(host='0.0.0.0', port=8000, debug=True)