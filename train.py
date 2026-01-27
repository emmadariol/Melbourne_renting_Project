import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from model_manager import ModelManager

def clean_data(df):
    """
    Applies data cleaning and preprocessing transformations based on the EDA findings.
    """
    print("Starting data cleaning process...")
    
    # --- Missing Value Imputation (Group-based) ---
    
    # Helper function to fill missing values with the median of the specific Suburb
    def fill_with_suburb_median(df, target_col, group_col='Suburb'):
        # Fill NaNs with the median of the suburb
        df[target_col] = df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))
        # Fallback: if a suburb has all NaNs, use the global median
        df[target_col] = df[target_col].fillna(df[target_col].median())
        return df

    # Impute BuildingArea and YearBuilt
    df = fill_with_suburb_median(df, 'BuildingArea')
    df = fill_with_suburb_median(df, 'YearBuilt')

    # 'Car' spots: assume NaN means 0 parking spots
    df['Car'] = df['Car'].fillna(0)

    # 'CouncilArea': fill with the mode (most frequent value) of the Suburb
    df['CouncilArea'] = df.groupby('Suburb')['CouncilArea'].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown")
    )
    df['CouncilArea'] = df['CouncilArea'].fillna("Unknown")

    # --- Data Type Conversion ---
    
    # Convert Date column to datetime objects
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    # Ensure integer columns are actually integers (round first to be safe)
    cols_to_int = ['Rooms', 'Bathroom', 'Car', 'Bedroom2', 'YearBuilt', 'Propertycount']
    for col in cols_to_int:
        # Fill any remaining NaNs with 0 before conversion to avoid errors
        df[col] = df[col].fillna(0).round().astype(int)

    # Drop rows where Price is missing (target variable quality)
    df = df.dropna(subset=['Price'])
    
    print(f"Cleaning complete. Dataset shape: {df.shape}")
    return df

def train():
    # Load Data
    try:
        df = pd.read_csv('melb_data.csv')
    except FileNotFoundError:
        print("Error: 'melb_data.csv' file not found.")
        return

    # Apply Custom Cleaning
    df_clean = clean_data(df)

    # Define Features for the Recommender System
    numerical_features = [
        'Rooms', 'Price', 'Distance', 'Bedroom2', 'Bathroom', 
        'Car', 'Landsize', 'BuildingArea', 'YearBuilt', 
        'Lattitude', 'Longtitude', 'Propertycount'
    ]
    
    # Categorical features that define the "style" and location macro-area
    categorical_features = ['Type', 'Regionname', 'Method']

    # Create the training dataset (X)
    X = df_clean[numerical_features + categorical_features].copy()

    # Build Scikit-Learn Pipeline
    
    # Numeric Transformer: Impute (safety net) -> Scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical Transformer: Impute -> OneHotEncode
    # handle_unknown='ignore' ensures the model doesn't crash if it sees a new category in production
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create the full Model Pipeline (Preprocessing + KNN Algorithm)
    # Metric 'cosine' is often better for high-dimensional mixed data than euclidean
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute'))
    ])


    # Validation step
    print("Performing Model Validation...")
    
    # Split data: 80% for training the temporary model, 20% for testing performance
    # This ensures we evaluate on unseen data before deploying the full model.
    X_train, X_test, y_train, y_test = train_test_split(X, df_clean['Price'], test_size=0.2, random_state=42)
    
    # Train a temporary pipeline on the training set only
    pipeline.fit(X_train)
    
    # Validation Logic: Predict prices for the Test Set
    
    # Transform the test set features using the preprocessor
    X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)
    
    # Find the K nearest neighbors for each house in the test set
    distances, indices = pipeline.named_steps['model'].kneighbors(X_test_transformed)
    
    # Calculate predicted price as the mean of neighbors' prices
    predicted_prices = []
    for neighbor_idx in indices:
        # Retrieve actual prices of the found neighbors from y_train
        mean_neighbor_price = y_train.iloc[neighbor_idx].mean()
        predicted_prices.append(mean_neighbor_price)
    
    # Calculate Key Performance Indicators (KPIs)
    mae = mean_absolute_error(y_test, predicted_prices)
    rmse = np.sqrt(mean_squared_error(y_test, predicted_prices)) # Penalizes large outliers
    r2 = r2_score(y_test, predicted_prices)                      # Accuracy score
    mape = mean_absolute_percentage_error(y_test, predicted_prices) * 100 # Error in percentage
    
    # Print Validation Report
    print(f"\n--- VALIDATION REPORT (Validation Set: {len(y_test)} samples) ---")
    print(f"   MAE  (Mean Absolute Error):      ${mae:,.0f}")
    print(f"   RMSE (Root Mean Squared Error):  ${rmse:,.0f}")
    print(f"   MAPE (Mean Percentage Error):    {mape:.2f}%")
    print(f"   R² Score (Variance Explained):   {r2:.4f}")
    print(f"---------------------------------------------------\n")


    # Train the Final Model on 100% of data
    print("Starting full model training on entire dataset...")

    # Fit the pipeline on the complete cleaned dataset
    pipeline.fit(X)
    print("Training complete.")

    # Save Artifacts using ModelManager
    manager = ModelManager()
    
    # We save a dictionary containing:
    # - The trained pipeline (to transform new inputs)
    # - The reference data (to retrieve house details for recommendations)
    # - The feature list (for validation)
    artifacts = {
        "pipeline": pipeline,
        "reference_data": df_clean, # Saving the clean DF is necessary to map indices back to real houses
        "features": numerical_features + categorical_features
    }
    
    version = manager.save_model(artifacts, note="Updated model with EDA-based cleaning and English comments")
    print(f"Model saved successfully. Version: {version}")

if __name__ == "__main__":
    train()