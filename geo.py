import osmnx as ox
from geopy.distance import great_circle
import pandas as pd
import warnings

# Suppress FutureWarning from OSMnx/Pandas interactions
warnings.filterwarnings("ignore")

class GeoService:
    def __init__(self):
        # 1. EXPANDED AMENITIES CONFIGURATION
        # Added 'default_dist' (in meters) to define specific relevance for each type
        self.amenities = {
            "lake":         {"tags": {"natural": "water", "water": "lake"}, "label": "Nature", "default_dist": 2000}, # Willing to travel far
            "park":         {"tags": {"leisure": "park", "landuse": "recreation_ground"}, "label": "Nature", "default_dist": 1000},
            "supermarket":  {"tags": {"shop": "supermarket"}, "label": "Shopping", "default_dist": 500}, # Must be close
            "train_station":{"tags": {"railway": "station"}, "label": "Transport", "default_dist": 1500},
            "tram_stop":    {"tags": {"railway": "tram_stop"}, "label": "Transport", "default_dist": 400},
            "bus_stop":     {"tags": {"highway": "bus_stop"}, "label": "Transport", "default_dist": 300}, # Must be very close
            "school":       {"tags": {"amenity": "school"}, "label": "Education", "default_dist": 1000},
            "hospital":     {"tags": {"amenity": "hospital"}, "label": "Health", "default_dist": 3000},
            "cafe":         {"tags": {"amenity": "cafe"}, "label": "Lifestyle", "default_dist": 500},
            "gym":          {"tags": {"leisure": "fitness_centre", "sport": "gym"}, "label": "Lifestyle", "default_dist": 800}
        }

    def check_nearby(self, lat, lon, feature_type, radius=None):
        """
        Single check: Is there a specific 'feature_type' nearby?
        If radius is None, it uses the specific default distance from config.
        """
        if feature_type not in self.amenities:
            return {"error": f"Unknown type. Available: {list(self.amenities.keys())}"}

        config = self.amenities[feature_type]
        tags = config["tags"]
        # Use provided radius or fallback to the specific default
        search_radius = radius if radius else config["default_dist"]
        
        try:
            pois = ox.features_from_point((lat, lon), tags=tags, dist=search_radius)
            
            if pois.empty:
                return {"found": False, "message": f"No {feature_type} found within {search_radius}m"}

            # Calculate distance to the closest one
            min_dist = pois.geometry.apply(
                lambda x: great_circle((lat, lon), (x.centroid.y, x.centroid.x)).meters
            ).min()

            return {
                "found": True, 
                "distance_meters": round(min_dist, 0),
                "threshold_used": search_radius,
                "message": f"Nearest {feature_type} is {round(min_dist, 0)}m away"
            }

        except Exception as e:
            return {"found": False, "error": str(e)}

    def scan_area(self, lat, lon, radius=None):
        """
        OPTIMIZED: Checks ALL amenities in one single API call.
        It downloads data based on the MAX distance needed (e.g. 2000m for Lake),
        but filters results based on specific limits (e.g. ignores Supermarket if > 500m).
        """
        all_tags = {}
        max_search_radius = 0

        # 1. Prepare tags and find the maximum radius needed for the API call
        for key, data in self.amenities.items():
            # Update max radius if this amenity needs a wider search
            if data["default_dist"] > max_search_radius:
                max_search_radius = data["default_dist"]

            for tag_key, tag_val in data["tags"].items():
                if tag_key not in all_tags:
                    all_tags[tag_key] = []
                if isinstance(tag_val, list):
                    all_tags[tag_key].extend(tag_val)
                else:
                    all_tags[tag_key].append(tag_val)

        report = {}
        
        try:
            # 2. Fetch everything at once using the LARGEST radius (One Network Request)
            all_pois = ox.features_from_point((lat, lon), tags=all_tags, dist=max_search_radius)
            
            if all_pois.empty:
                return {"status": "empty", "data": {}}

            # 3. Process the results locally
            for key, data in self.amenities.items():
                tags = data["tags"]
                limit_dist = data["default_dist"] # The specific limit for this amenity
                
                # Filter the GeoDataFrame for this specific amenity
                mask = pd.Series([False] * len(all_pois), index=all_pois.index)
                for t_key, t_val in tags.items():
                    if t_key in all_pois.columns:
                        mask |= all_pois[t_key].isin([t_val] if isinstance(t_val, str) else t_val)
                
                subset = all_pois[mask]
                
                if not subset.empty:
                    dist = subset.geometry.apply(
                        lambda x: great_circle((lat, lon), (x.centroid.y, x.centroid.x)).meters
                    ).min()
                    
                    # LOGIC CHANGE: Only report if it is within its SPECIFIC limit
                    if dist <= limit_dist:
                        report[key] = round(dist, 0)
                    else:
                        report[key] = None # Found, but too far for this type of service
                else:
                    report[key] = None # Not found

            return {"status": "success", "data": report}

        except Exception as e:
            return {"status": "error", "message": str(e)}

# Test block
if __name__ == "__main__":
    geo = GeoService()
    
    # Coordinates for Flinders Street Station (Central Melbourne)
    test_lat, test_lon = -37.8183, 144.9671
    
    print("--- Testing Single Check (Gym) ---")
    print(geo.check_nearby(test_lat, test_lon, "gym"))
    
    print("\n--- Testing Full Area Scan (Smart Distances) ---")
    # This will check for Lakes up to 2km, but Supermarkets only up to 500m
    report = geo.scan_area(test_lat, test_lon)
    print(report)