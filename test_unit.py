import pytest
import pandas as pd
import numpy as np
from train import clean_data


@pytest.fixture
def dirty_data():
    """
    Creates a temporary DataFrame with 'dirty' data (missing values, NaNs)
    to verify if the cleaning pipeline handles them correctly.
    """
    return pd.DataFrame({
        'Suburb': ['Richmond', 'South Yarra', 'Richmond'], 
        # 'Price' has one missing value (NaN), which is the target variable.
        'Price': [1000000.0, np.nan, 500000.0],  
        'Date': ['01/01/2020', '01/01/2020', '01/01/2020'],
        # 'Rooms' has a missing value.
        'Rooms': [3, 2, np.nan],                
        'Bathroom': [1, 1, 1], 
        'Bedroom2': [3, 2, np.nan],
        # 'Car' spots has a missing value.
        'Car': [1, np.nan, 2],                  
        'BuildingArea': [100, np.nan, 100],     
        'YearBuilt': [2000.0, 1990, np.nan],      
        # 'CouncilArea' has a missing value to test categorical imputation.
        'CouncilArea': ['Yarra', 'Stonnington', np.nan], 
        'Lattitude': [-37.8, -37.8, -37.8],
        'Longtitude': [144.9, 144.9, 144.9],
        'Propertycount': [4000, 4000, 4000],
        'Distance': [2.5, 3.0, 2.5]
    })

# ==========================================
# UNIT TESTS
# ==========================================

def test_clean_data_drops_missing_price(dirty_data):
    """
    Test Case 1: Target Variable Integrity.
    
    The 'Price' column is our target variable. We cannot train a model 
    on rows where the target is missing. This test verifies that 
    rows with NaN Price are dropped from the dataset.
    """
    # Apply cleaning function
    df_clean = clean_data(dirty_data)
    
    # Assertions
    # The row with NaN Price should be removed (3 original rows -> 2 remaining)
    assert len(df_clean) == 2, "Rows with missing Price should be dropped"
    
    # Verify no NaN values remain in Price column
    assert df_clean['Price'].isna().sum() == 0, "Price column should not contain NaNs"

def test_imputation_logic(dirty_data):
    """
    Test Case 2: Numerical Feature Imputation.
    
    Features like 'Car' (parking spots) and 'Rooms' are essential.
    This test verifies that missing numerical values are filled (imputed)
    and converted to the correct data type (e.g., Integer).
    """
    df_clean = clean_data(dirty_data)
    
    # 'Car' should be filled (e.g., with 0 or median), never NaN
    assert df_clean['Car'].isna().sum() == 0, "Car column should be imputed"
    
    # 'Rooms' should be filled and converted to a safe numeric type (Integer)
    # Pandas can use int32 or int64, so we check generally for integer type
    assert pd.api.types.is_integer_dtype(df_clean['Rooms']), "Rooms should be converted to integer"

    # Ensures dates are like 1990, not 1990.0
    assert df_clean['YearBuilt'].isna().sum() == 0
    assert pd.api.types.is_integer_dtype(df_clean['YearBuilt']), "YearBuilt should be converted to integer"

def test_council_area_imputation(dirty_data):
    """
    Test Case 3: Categorical Feature Imputation.
    
    'CouncilArea' is a categorical string. If missing, it should be 
    inferred from the Suburb (mode) or set to a default value.
    It should definitely NOT be empty or None.
    """
    df_clean = clean_data(dirty_data)
    
    # Check for NaNs
    assert df_clean['CouncilArea'].isna().sum() == 0, "CouncilArea should be imputed"
    
    # Check that logic inferred a value (not left as 'Unknown' if neighbor exists)
    # Note: In our mock data, 'Richmond' appears twice, so it might infer 'Yarra'
    assert "Unknown" not in df_clean['CouncilArea'].values, "Categorical imputation failed to infer value"