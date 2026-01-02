import osmnx as ox
from geopy.distance import great_circle
import pandas as pd
import warnings

# Suppress FutureWarning from OSMnx/Pandas interactions
warnings.filterwarnings("ignore")

class GeoService:
    def __init__(self):
        # 1. EXPANDED AMENITIES CONFIGURATION
        # We categorize them to make reporting easier.
        # This list includes supermarkets, schools, lakes, and more.
        self.amenities = {
            "lake":         {"tags": {"natural": "water", "water": "lake"}, "label": "Nature"},
            "park":         {"tags": {"leisure": "park", "landuse": "recreation_ground"}, "label": "Nature"},
            "supermarket":  {"tags": {"shop": "supermarket"}, "label": "Shopping"},
            "train_station":{ "tags": {"railway": "station"}, "label": "Transport"},
            "tram_stop":    {"tags": {"railway": "tram_stop"}, "label": "Transport"}, # Melbourne essential!
            "bus_stop":     {"tags": {"highway": "bus_stop"}, "label": "Transport"},
            "school":       {"tags": {"amenity": "school"}, "label": "Education"},
            "hospital":     {"tags": {"amenity": "hospital"}, "label": "Health"},
            "cafe":         {"tags": {"amenity": "cafe"}, "label": "Lifestyle"},
            "gym":          {"tags": {"leisure": "fitness_centre", "sport": "gym"}, "label": "Lifestyle"}
        }

    def check_nearby(self, lat, lon, feature_type, radius=1000):
        """
        Single check: Is there a specific 'feature_type' nearby?
        Good for specific user queries like "Is there a gym?"
        """
        if feature_type not in self.amenities:
            return {"error": f"Unknown type. Available: {list(self.amenities.keys())}"}

        tags = self.amenities[feature_type]["tags"]
        
        try:
            pois = ox.features_from_point((lat, lon), tags=tags, dist=radius)
            
            if pois.empty:
                return {"found": False, "message": f"No {feature_type} found within {radius}m"}

            # Calculate distance to the closest one
            # Use geometry centroid for safety (handles polygons like lakes/parks)
            min_dist = pois.geometry.apply(
                lambda x: great_circle((lat, lon), (x.centroid.y, x.centroid.x)).meters
            ).min()

            return {
                "found": True, 
                "distance_meters": round(min_dist, 0),
                "message": f"Nearest {feature_type} is {round(min_dist, 0)}m away"
            }

        except Exception as e:
            return {"found": False, "error": str(e)}

    def scan_area(self, lat, lon, radius=1000):
        """
        OPTIMIZED: Checks ALL amenities in one single API call.
        Great for generating a 'House Report Card'.
        """
        # 1. Combine all tags into one big query to save time
        all_tags = {}
        for key, data in self.amenities.items():
            for tag_key, tag_val in data["tags"].items():
                if tag_key not in all_tags:
                    all_tags[tag_key] = []
                # Handle single strings vs lists in tags
                if isinstance(tag_val, list):
                    all_tags[tag_key].extend(tag_val)
                else:
                    all_tags[tag_key].append(tag_val)

        report = {}
        
        try:
            # 2. Fetch everything at once (One Network Request)
            # This is much faster than calling check_nearby 10 times
            all_pois = ox.features_from_point((lat, lon), tags=all_tags, dist=radius)
            
            if all_pois.empty:
                # Return empty report if nothing is found
                return {"status": "success", "data": {k: None for k in self.amenities}}

            # 3. Process the results locally
            # We iterate through our defined amenities and filter the downloaded data
            for key, data in self.amenities.items():
                tags = data["tags"]
                
                # Filter the GeoDataFrame for this specific amenity
                # This logic checks if the row matches ANY of the tags for this amenity
                mask = pd.Series([False] * len(all_pois), index=all_pois.index)
                for t_key, t_val in tags.items():
                    if t_key in all_pois.columns:
                        mask |= all_pois[t_key].isin([t_val] if isinstance(t_val, str) else t_val)
                
                subset = all_pois[mask]
                
                if not subset.empty:
                    dist = subset.geometry.apply(
                        lambda x: great_circle((lat, lon), (x.centroid.y, x.centroid.x)).meters
                    ).min()
                    report[key] = round(dist, 0)
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
    
    print("--- Testing Full Area Scan ---")
    report = geo.scan_area(test_lat, test_lon)
    print(report)