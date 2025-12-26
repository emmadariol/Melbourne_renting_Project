from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from geo import GeoService  # Ensure geo.py is in the same folder

app = FastAPI(title="Melbourne Housing Recommender")

# Load Artifacts
try:
    artifacts = joblib.load("model_artifacts.pkl")
    model_pipeline = artifacts["pipeline"]
    db = artifacts["database"]
    NUM_FEATS = artifacts["numeric_features"]
    print("Artifacts loaded.")
except FileNotFoundError:
    raise RuntimeError("Run train.py first!")

geo_service = GeoService()

# Pydantic Model matches the Training columns
class UserPreferences(BaseModel):
    Rooms: int = Field(3, description="Number of rooms")
    Price: float = Field(1000000, description="Target Price")
    Distance: float = Field(10.0, description="Distance from CBD")
    Bathroom: int = Field(2, description="Number of Bathrooms")
    Car: int = Field(1, description="Number of Car spots")
    Landsize: float = Field(500.0, description="Land Size")
    BuildingArea: float = Field(150.0, description="Building Size")
    Propertycount: float = Field(4000.0, description="Suburb density (Property count)")
    YearBuilt: float = Field(2000.0, description="Approximate Year Built")
    
    # Categorical
    Type: str = Field("h", description="h=house, u=unit, t=townhouse")
    Regionname: str = Field("Southern Metropolitan", description="General Region")

@app.post("/recommend")
def recommend_house(prefs: UserPreferences):
    input_df = pd.DataFrame([prefs.dict()])
    
    # Ensure columns match training order exactly
    # We filter input_df to only columns that were in training
    # (This handles cases where API input might have extras)
    
    try:
        # Transform input
        processed_input = model_pipeline.named_steps['preprocessor'].transform(input_df)
        
        # Inference
        distances, indices = model_pipeline.named_steps['knn'].kneighbors(processed_input)
        
        # Retrieve results
        results = []
        for idx in indices[0]:
            house = db.iloc[idx].to_dict()
            # Clean NaNs for JSON
            house = {k: (v if pd.notna(v) else None) for k, v in house.items()}
            results.append(house)
            
        return {"recommendations": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/amenities")
def check_amenities(lat: float, lon: float, amenity: str):
    return geo_service.check_nearby(lat, lon, amenity)