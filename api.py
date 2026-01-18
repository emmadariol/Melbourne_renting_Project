from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import pandas as pd
from geo import GeoService  # Ensure geo.py is in the same folder

app = FastAPI(title="Melbourne Housing Recommender")

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

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
    
    # Optional: Required nearby amenities
    required_amenities: Optional[List[str]] = Field(None, description="List of required nearby amenities")

@app.post("/recommend")
def recommend_house(prefs: UserPreferences):
    # Remove required_amenities from the dataframe (it's not a model feature)
    prefs_dict = prefs.dict()
    required_amenities = prefs_dict.pop('required_amenities', None)
    
    input_df = pd.DataFrame([prefs_dict])
    
    # Ensure columns match training order exactly
    # We filter input_df to only columns that were in training
    # (This handles cases where API input might have extras)
    
    try:
        # Transform input
        processed_input = model_pipeline.named_steps['preprocessor'].transform(input_df)
        
        # Inference - get more results if filtering is needed
        n_neighbors = 50 if required_amenities else 5
        distances, indices = model_pipeline.named_steps['knn'].kneighbors(processed_input, n_neighbors=n_neighbors)
        
        # Retrieve results
        results = []
        checked_count = 0
        max_checks = n_neighbors  # Limit how many we check to avoid timeout
        
        for idx in indices[0]:
            if checked_count >= max_checks:
                break
                
            house = db.iloc[idx].to_dict()
            # Clean NaNs for JSON
            house = {k: (v if pd.notna(v) else None) for k, v in house.items()}
            
            # Filter by required amenities if specified
            if required_amenities and house.get('Lattitude') and house.get('Longtitude'):
                checked_count += 1
                meets_requirements = True
                
                for amenity in required_amenities:
                    if amenity in geo_service.amenities:
                        try:
                            check_result = geo_service.check_nearby(
                                house['Lattitude'], 
                                house['Longtitude'], 
                                amenity
                            )
                            if not check_result.get('found', False):
                                meets_requirements = False
                                break
                        except Exception as e:
                            # Skip this house if geo check fails
                            print(f"Geo check failed for {amenity}: {e}")
                            meets_requirements = False
                            break
                
                if meets_requirements:
                    results.append(house)
            else:
                results.append(house)
            
            # Stop once we have enough results
            if len(results) >= 5:
                break
        
        # If we didn't find enough results with filters, inform the user
        if required_amenities and len(results) < 5:
            print(f"Warning: Only found {len(results)} houses matching amenity requirements after checking {checked_count} candidates")
            
        return {"recommendations": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/house_report")
def get_house_report(lat: float, lon: float):
    """
    Returns distances to all key amenities (Schools, Transport, etc.)
    """
    return geo_service.scan_area(lat, lon)