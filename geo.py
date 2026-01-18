import osmnx as ox
from geopy.distance import great_circle
import pandas as pd
import warnings
from functools import lru_cache
import hashlib
import json

# Suppress FutureWarning from OSMnx/Pandas interactions
warnings.filterwarnings("ignore")

class GeoService:
    def __init__(self):
        # 1. EXPANDED AMENITIES CONFIGURATION
        # We categorize them to make reporting easier.
        # This list includes supermarkets, schools, lakes, and more.
        # Each service now has its own sensible radius (in meters)
        self.amenities = {
            "lake":         {"tags": {"natural": "water", "water": "lake"}, "label": "Nature", "radius": 3000},
            "park":         {"tags": {"leisure": "park", "landuse": "recreation_ground"}, "label": "Nature", "radius": 1000},
            "supermarket":  {"tags": {"shop": "supermarket"}, "label": "Shopping", "radius": 500},
            "train_station":{ "tags": {"railway": "station"}, "label": "Transport", "radius": 1500},
            "tram_stop":    {"tags": {"railway": "tram_stop"}, "label": "Transport", "radius": 400}, # Melbourne essential!
            "bus_stop":     {"tags": {"highway": "bus_stop"}, "label": "Transport", "radius": 300},
            "school":       {"tags": {"amenity": "school"}, "label": "Education", "radius": 1500},
            "hospital":     {"tags": {"amenity": "hospital"}, "label": "Health", "radius": 3000},
            "cafe":         {"tags": {"amenity": "cafe"}, "label": "Lifestyle", "radius": 500},
            "gym":          {"tags": {"leisure": "fitness_centre", "sport": "gym"}, "label": "Lifestyle", "radius": 1000}
        }
        # Simple in-memory cache for repeated queries
        self._cache = {}
    
    def _get_cache_key(self, lat, lon, feature_type, radius):
        """Generate a cache key for a query"""
        # Round coordinates to 5 decimal places (~1 meter precision)
        return f"{round(lat, 5)}_{round(lon, 5)}_{feature_type}_{radius}"
    
    def check_nearby(self, lat, lon, feature_type, radius=None):
        """
        Single check: Is there a specific 'feature_type' nearby?
        Good for specific user queries like "Is there a gym?"
        If radius is not specified, uses the service-specific default radius.
        """
        if feature_type not in self.amenities:
            return {"error": f"Unknown type. Available: {list(self.amenities.keys())}"}

        tags = self.amenities[feature_type]["tags"]
        
        # Use service-specific radius if not provided
        if radius is None:
            radius = self.amenities[feature_type]["radius"]
        
        # Check cache first
        cache_key = self._get_cache_key(lat, lon, feature_type, radius)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            pois = ox.features_from_point((lat, lon), tags=tags, dist=radius)
            
            if pois.empty:
                result = {"found": False, "message": f"No {feature_type} found within {radius}m"}
                self._cache[cache_key] = result
                return result

            # Calculate distance to the closest one
            # Use geometry centroid for safety (handles polygons like lakes/parks)
            min_dist = pois.geometry.apply(
                lambda x: great_circle((lat, lon), (x.centroid.y, x.centroid.x)).meters
            ).min()

            result = {
                "found": True, 
                "distance_meters": round(min_dist, 0),
                "message": f"Nearest {feature_type} is {round(min_dist, 0)}m away"
            }
            self._cache[cache_key] = result
            return result

        except Exception as e:
            result = {"found": False, "error": str(e)}
            return result

    def scan_area(self, lat, lon):
        """
        OPTIMIZED: Checks ALL amenities using their service-specific radii.
        Great for generating a 'House Report Card'.
        Each service type uses its own appropriate search radius.
        """
        report = {}
        
        try:
            # Process each amenity with its specific radius
            for key, data in self.amenities.items():
                tags = data["tags"]
                radius = data["radius"]
                
                # Fetch POIs for this specific amenity with its own radius
                pois = ox.features_from_point((lat, lon), tags=tags, dist=radius)
                
                if not pois.empty:
                    dist = pois.geometry.apply(
                        lambda x: great_circle((lat, lon), (x.centroid.y, x.centroid.x)).meters
                    ).min()
                    report[key] = round(dist, 0)
                else:
                    report[key] = None # Not found within the service-specific radius

            return {"status": "success", "data": report}

        except Exception as e:
            return {"status": "error", "message": str(e)}

# Test block
if __name__ == "__main__":
    geo = GeoService()
    
    # Coordinates for Flinders Street Station (Central Melbourne)
    test_lat, test_lon = -37.8183, 144.9671
    
    print("--- Testing Full Area Scan ---")
    report = geo.scan_area(test_lat, test_lon)
    print(report)