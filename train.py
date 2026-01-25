import pandas as pd
import numpy as np
import joblib
import uuid
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from model_manager import ModelManager

# 1. Load Data
try:
    df = pd.read_csv('melb_data.csv')
    print("Dataset loaded.")
except FileNotFoundError:
    print("Error: melb_data.csv not found.")
    exit()

# Generate Unique IDs
df['HouseID'] = [str(uuid.uuid4()) for _ in range(len(df))]

# 2. Feature Selection
numeric_features = [
    'Rooms', 'Price', 'Distance', 'Bathroom', 'Car', 
    'Landsize', 'BuildingArea', 'Propertycount'
]

# [UPDATE] Added CouncilArea here
categorical_features = ['Type', 'Regionname', 'CouncilArea']

# 3. Data Cleaning
# Ensure we drop rows where CouncilArea is missing too
df = df.dropna(subset=['Price', 'Lattitude', 'Longtitude', 'Regionname', 'Type', 'CouncilArea'])

if 'YearBuilt' in df.columns:
    numeric_features.append('YearBuilt')

# 4. Preprocessing Pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 5. Build Model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('knn', NearestNeighbors(n_neighbors=5, metric='euclidean'))
])

print(f"Training on features: {numeric_features + categorical_features}")
model.fit(df)

# 6. Save Artifacts 
artifacts = {
    "pipeline": model,
    "database": df.reset_index(drop=True),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features
}

manager = ModelManager()
manager.save_model(artifacts, note="Initial training from train.py")