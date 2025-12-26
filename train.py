import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 1. Load Data
try:
    df = pd.read_csv('melb_data.csv')
    print("Dataset loaded.")
except FileNotFoundError:
    print("Error: melb_data.csv not found.")
    exit()

# 2. Feature Selection
# We focus on variables that define the "Physical House" and "Location"
# Excluded: Method, SellerG, Date (Sale metadata), Bedroom2 (Redundant)
numeric_features = [
    'Rooms', 'Price', 'Distance', 'Bathroom', 'Car', 
    'Landsize', 'BuildingArea', 'Propertycount'
]
categorical_features = ['Type', 'Regionname'] # 'CouncilArea' is too high cardinality, Regionname is better

# 3. Data Cleaning
# Drop rows where target critical info is missing
df = df.dropna(subset=['Price', 'Lattitude', 'Longtitude', 'Regionname', 'Type'])

# Handle "YearBuilt": It has many NaNs. We'll impute, but sometimes it's better to drop if too many are missing.
# For this recommender, we will keep it but treat it carefully.
# Actually, given your variable list, I'll add YearBuilt to numeric if it exists, otherwise skip.
if 'YearBuilt' in df.columns:
    numeric_features.append('YearBuilt')

# 4. Preprocessing Pipeline
# Numeric: Fill missing values with Median -> Scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical: OneHotEncode (Handles 'h', 'u', 't', 'dev site', etc.)
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

joblib.dump(artifacts, "model_artifacts.pkl")
print("Model updated and saved.")