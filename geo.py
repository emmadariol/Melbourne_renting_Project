import osmnx as ox
from geopy.distance import great_circle

class GeoService:
    def __init__(self):
        # Configuration for OSM tags
        self.amenities = {
            "lake": {"natural": "water", "water": "lake"},
            "supermarket": {"shop": "supermarket"},
            "park": {"leisure": "park"}
        }

    def check_nearby(self, lat, lon, feature_type, radius=1000):
        """
        Checks if a feature is within 'radius' meters of (lat, lon).
        """
        if feature_type not in self.amenities:
            return {"error": "Unknown feature type"}

        tags = self.amenities[feature_type]
        
        try:
            # 1. Get features from OSM within the radius
            # Note: In production, you would cache this or use a PostGIS DB.
            pois = ox.features_from_point((lat, lon), tags=tags, dist=radius)
            
            if pois.empty:
                return {"found": False, "message": f"No {feature_type} found within {radius}m"}

            # 2. Calculate distance to the closest one
            # We use the centroid of the found shape (polygon) to calculate distance
            min_dist = pois.geometry.apply(
                lambda x: great_circle((lat, lon), (x.centroid.y, x.centroid.x)).meters
            ).min()

            return {
                "found": True, 
                "distance_meters": round(min_dist, 2),
                "message": f"Nearest {feature_type} is {round(min_dist, 2)}m away"
            }

        except Exception as e:
            # Fallback for API errors or timeouts
            return {"found": False, "error": str(e)}

# Test block
if __name__ == "__main__":
    geo = GeoService()
    # Test with a coordinate in Melbourne
    print(geo.check_nearby(-37.8136, 144.9631, "supermarket"))