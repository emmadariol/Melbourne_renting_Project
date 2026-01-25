import joblib
import json
import os
import time
import shutil

class ModelManager:
    def __init__(self, models_dir="models", registry_file="registry.json"):
        # Absolute path ensures consistency across different script locations
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(base_path, models_dir)
        self.registry_path = os.path.join(self.models_dir, registry_file)
        
        # Ensure directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize registry if it doesn't exist
        if not os.path.exists(self.registry_path):
            self._save_registry({"current": None, "history": []})

    def _load_registry(self):
        try:
            with open(self.registry_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"current": None, "history": []}

    def _save_registry(self, data):
        # Atomic write to prevent corruption
        temp_path = self.registry_path + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=4)
        shutil.move(temp_path, self.registry_path)

    def save_model(self, artifacts, note=""):
        """Saves a new model version and updates the registry."""
        timestamp = int(time.time())
        version_id = f"model_v{timestamp}"
        filename = f"{version_id}.pkl"
        filepath = os.path.join(self.models_dir, filename)

        # 1. Save the Artifact File
        joblib.dump(artifacts, filepath)
        print(f"✅ Saved artifact: {filepath}")

        # 2. Update Registry
        reg = self._load_registry()
        reg["current"] = filename
        reg["history"].append({
            "version": version_id,
            "filename": filename,
            "timestamp": timestamp,
            "note": note
        })
        self._save_registry(reg)
        print(f"✅ Registry updated. Current is now: {filename}")
        return version_id

    def load_current_model(self):
        """Loads the model pointed to by 'current' in registry."""
        reg = self._load_registry()
        current_file = reg.get("current")

        if not current_file:
            raise FileNotFoundError(f"No active model found in registry at {self.registry_path}")

        filepath = os.path.join(self.models_dir, current_file)
        
        if not os.path.exists(filepath):
             raise FileNotFoundError(f"Registry points to {filepath}, but file is missing.")
             
        print(f"🔄 Loading model: {filepath}")
        return joblib.load(filepath)