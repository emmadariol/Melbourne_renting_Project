from locust import HttpUser, task, between
import random

class MelbourneHousingUser(HttpUser):
    wait_time = between(1, 3) # Simulate realistic user thinking time

    @task(3)
    def get_health(self):
        """Monitor system uptime and house count."""
        self.client.get("/health")

    @task(5)
    def test_recommendation(self):
        """Simulate users searching for recommendations."""
        payload = {
            "Rooms": random.randint(1, 5),
            "Price": random.uniform(500000, 2000000),
            "Distance": random.uniform(1.0, 20.0),
            "Type": random.choice(["h", "u", "t"]),
            "Regionname": "Southern Metropolitan"
        }
        self.client.post("/recommend", json=payload)

    @task(2)
    def test_search(self):
        """Simulate agents searching for specific addresses."""
        queries = ["Richmond", "Abbotsford", "St Kilda"]
        self.client.get(f"/agent/search?query={random.choice(queries)}")

    @task(1)
    def test_geo_report(self):
        """Simulate intensive geospatial scans."""
        # Testing near the CBD
        self.client.get("/house_report?lat=-37.8136&lon=144.9631")