from flask import Flask, request, jsonify
import pandas as pd
import uuid
import os
from pydantic import BaseModel, Field, ValidationError
from geo import GeoService
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from model_manager import ModelManager

app = Flask(__name__)

# --- Configuration ---
geo_service = GeoService()
geocoder = Nominatim(user_agent="melbourne_housing_app")
manager = ModelManager()

# Global State
model_pipeline = None
database = None
preprocessor = None
features_list = []

# --- Data Models ---
class UserPreferences(BaseModel):
    # Numerical features
    Rooms: int = Field(3)
    Price: float = Field(1000000)
    Distance: float = Field(10.0)
    Bedroom2: int = Field(3)
    Bathroom: int = Field(2)
    Car: int = Field(1)
    Landsize: float = Field(500.0)
    BuildingArea: float = Field(150.0)
    YearBuilt: float = Field(2000.0)
    Propertycount: float = Field(4000.0)
    Lattitude: float = Field(-37.8)
    Longtitude: float = Field(144.9)
    
    # Categorical features
    Type: str = Field("h")
    Regionname: str = Field("Southern Metropolitan")
    Method: str = Field("S")

class NewHouse(UserPreferences):
    Suburb: str = Field(..., description="Suburb Name")
    Address: str = Field(..., description="Street Address")
    SellerG: str = Field("Private", description="Seller Agency")
    CouncilArea: str = Field("Unknown", description="Council Area")

class UpdateHouse(NewHouse):
    HouseID: str = Field(..., description="Unique ID is required for updates")

# --- Core Logic ---

def load_system():
    global model_pipeline, database, preprocessor, features_list
    try:
        # Load via Manager
        artifacts = manager.load_current_model()
        
        model_pipeline = artifacts["pipeline"]
        
        # Handle different naming conventions
        database = artifacts.get("reference_data")
        if database is None:
            database = artifacts.get("database")
            
        if database is None:
            raise KeyError("Neither 'reference_data' nor 'database' found in model artifacts.")

        # Load feature list (new train.py saves combined 'features')
        features_list = artifacts.get("features", [])
        if not features_list:
             # Fallback for older models
             features_list = artifacts.get("numeric_features", []) + artifacts.get("categorical_features", [])

        preprocessor = model_pipeline.named_steps['preprocessor']
        
        # Ensure IDs exist in the loaded data
        if 'HouseID' not in database.columns:
            print("⚠️ Adding missing HouseIDs to loaded data...")
            database['HouseID'] = [str(uuid.uuid4()) for _ in range(len(database))]
            # We don't autosave here to avoid overwriting the model file unnecessarily on boot
            
        print(f"✅ System Loaded Successfully. {len(database)} houses in database.")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR loading system: {e}")
        print("Tip: Run 'python train.py' to generate a valid model file first.")
        # We do not exit here so the app can start (and show health=uninitialized)

def save_system(note="Auto-update"):
    """Saves the current state of the database and model."""
    if model_pipeline is None or database is None:
        print("⚠️ Cannot save system: State is not initialized.")
        return

    artifacts = {
        "pipeline": model_pipeline,
        "reference_data": database, # Saving with the new consistent key
        "features": features_list
    }
    # Save via Manager
    manager.save_model(artifacts, note=note)

def retrain_model(note="Online Learning"):
    """Refits the KNN model on the current database (Online Learning)."""
    if model_pipeline is None: return

    print("🔄 Retraining KNN model on updated data...")
    try:
        # Transform current database using the pre-trained preprocessor
        X_transformed = preprocessor.transform(database)
        
        # Refit only the KNN part
        knn = model_pipeline.named_steps['model']
        knn.fit(X_transformed)
        
        # Update pipeline step
        model_pipeline.steps[-1] = ('model', knn)
        
        save_system(note=note)
        print("✅ Retraining complete and saved.")
    except Exception as e:
        print(f"⚠️ Error during retraining: {e}")

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
        # Remove fields that might be sent but not in Pydantic model
        _ = data.pop('required_amenities', None)
        
        prefs = UserPreferences(**data)
        
        # Create DataFrame for input, ensuring columns match training features
        input_df = pd.DataFrame([prefs.dict()])
        
        # Ensure all columns required by preprocessor exist (fill missing with defaults/NaN)
        # The preprocessor expects columns in specific order/presence
        # We rely on the Pydantic model covering the necessary features used in train.py
        
        processed_input = preprocessor.transform(input_df)
        
        # KNN search
        knn = model_pipeline.named_steps['model']
        distances, indices = knn.kneighbors(processed_input)
        
        results = []
        for idx in indices[0]:
            if idx < len(database):
                house = database.iloc[idx].to_dict()
                # Clean NaNs for JSON response
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
    
    # Simple string match on Address or Suburb
    mask = (database['Address'].astype(str).str.lower().str.contains(query, na=False) | 
            database['Suburb'].astype(str).str.lower().str.contains(query, na=False))
            
    matches = database[mask].head(20).to_dict(orient='records')
    clean_matches = [{k: (v if pd.notna(v) else None) for k, v in h.items()} for h in matches]
    return jsonify({"results": clean_matches})

@app.route('/add_house', methods=['POST'])
def add_house():
    global database
    if database is None: return jsonify({"error": "System not initialized"}), 503
    try:
        data = request.json
        new_house_data = NewHouse(**data).dict()
        new_house_data['HouseID'] = str(uuid.uuid4())
        
        # Align columns (fill missing columns in new_house_data with defaults or NaN)
        # This ensures the new row matches the database schema
        for col in database.columns:
            if col not in new_house_data:
                new_house_data[col] = None
                
        database = pd.concat([database, pd.DataFrame([new_house_data])], ignore_index=True)
        retrain_model(note=f"Added house {new_house_data.get('Address', 'Unknown')}")
        return jsonify({"message": "House added", "id": new_house_data['HouseID']})
    except ValidationError as e: return jsonify({"error": e.errors()}), 400

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

if __name__ == '__main__':
    load_system()
    app.run(host='0.0.0.0', port=8000, debug=True)