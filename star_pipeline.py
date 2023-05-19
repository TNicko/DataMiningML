# ----------------------------------------------------------------------------------------------------
# Description: This file contains the constants and functions used specifically for processing the Star dataset.
# ----------------------------------------------------------------------------------------------------

import numpy as np
from processing import fill_missing_numbers, convert_int_to_float

# TODO
# Remove first ID row as it is useless ???
# Convert an ID row to catgorical???

# For now i have removed both id rows

def star_pipeline(data: np.ndarray) -> np.ndarray:
    """Converts the data to the correct data types and fills in missing values."""

    for column, dtype in dict(data.dtype.fields).items():
        dtype = dtype[0]
        if dtype.char in {'S', 'U'}:
            # Clear all whitespace from string columns
            data[:][column] = np.char.strip(data[:][column])

        data = convert_int_to_float(data) # convert int cols to float
        
        if dtype == np.float64:
            # Fill missing values in all columns containing numbers
            data = fill_missing_numbers(data, column)
    
    # # Encode categorical columns
    # categorical_cols = ['fiber_ID']
    # data = encode_categorical_data(data, categorical_cols)
    
    return data



