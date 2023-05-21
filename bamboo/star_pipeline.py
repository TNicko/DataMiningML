# ----------------------------------------------------------------------------------------------------
# Description: This file contains the constants and functions used specifically for processing the Star dataset.
# ----------------------------------------------------------------------------------------------------

import numpy as np
from bamboo.processing import encode_categorical_data, fill_missing_numbers, convert_int_to_float, get_categorical_columns, fill_categorical_missing

# TODO
# Remove first ID row as it is useless ???
# Convert an ID row to catgorical???

# For now i have removed both id rows

def star_pipeline(data: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Converts the data to the correct data types and fills in missing values."""
    np.random.seed(seed)

    for column, dtype in dict(data.dtype.fields).items():
        dtype = dtype[0]
        if dtype.char in {'S', 'U'}:
            # Clear all whitespace from string columns
            data[:][column] = np.char.strip(data[:][column])

        data = convert_int_to_float(data) # convert int cols to float
        
        if dtype == np.float64:
            # Fill missing values in all columns containing numbers
            data = fill_missing_numbers(data, column)
    
    # Get cateforical columns
    categorical_cols = get_categorical_columns(data)

    # Fill missing values in categorical columns
    for column in categorical_cols:
        data = fill_categorical_missing(data, column)
    
    # Encode categorical columns
    data = encode_categorical_data(data, categorical_cols)

    
    return data



