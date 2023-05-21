# ----------------------------------------------------------------------------------------------------
# Description : This file contains the constants and functions used specifically for processing the GWP dataset.
# ----------------------------------------------------------------------------------------------------

import numpy as np
from datetime import datetime
from bamboo.processing import fill_missing_numbers, encode_categorical_data, split_date_column, convert_dates, convert_int_to_float, get_categorical_columns, fill_categorical_missing


def gwp_pipeline(data: np.ndarray, seed: int | None = None) -> np.ndarray:
    """
    Applies a pipeline of transformations to the GWP structured numpy array for data preprocessing.
    """
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

    data = convert_dates(data)        # convert dates to datetime objects
    data = process_gwp_temporal(data) # fill missing dates and days
    data = process_quarter(data)      # fill missing quarters

    # Split date column into multiple feature columns
    data = split_date_column(data)

    # Get cateforical columns
    categorical_cols = get_categorical_columns(data)

    # Fill missing values in categorical columns
    for column in categorical_cols:
        data = fill_categorical_missing(data, column)
    
    # Encode categorical columns
    data = encode_categorical_data(data, categorical_cols)

    return data


# ----------------------------------------------------------------------------------------------------
# Temporal data has a unique structure and inherent ordering that's not present in other types of data. 
# This means that the techniques we typically use for handling missing data in numerical or categorical 
# features might not be appropriate for temporal data. Therefore, in this case, to fill in the missing 
# data with the correct temporal values the two functions below are used. 
# ----------------------------------------------------------------------------------------------------

def process_gwp_temporal(data: np.ndarray) -> np.ndarray:
    """
    Processes and fill in missing temporal information of a numpy structured array.

    The function fills the missing 'date' and 'day' fields by using the existing temporal information.
    If both 'date' and 'day' fields exist, the function just stores the values.
    If only 'date' is missing, it is calculated by using the 'day' field and the previous date.
    If only 'day' is missing, it is calculated by using the 'date' field.
    If both 'date' and 'day' are missing, they are calculated by using the previous date and day and the 'quarter' field.
    """
    previous_date = None
    previous_day = None

    day_names = np.array(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    for i, row in enumerate(data):
        date = row['date']
        day = row['day']

        # 1. Both date and day exist
        if not np.isnat(date) and day:
            previous_date = date
            previous_day = day

        # 2. Only date is missing
        elif np.isnat(date) and day:
            if previous_day == day:
                row['date'] = previous_date
            else:
                # Check if the day is a Saturday and add two days since Friday is skipped.
                date = previous_date + np.timedelta64(1, 'D')
                if day == "Saturday":
                    date += np.timedelta64(1, 'D')
                    row['date'] = date
                    previous_date = date
                    previous_day = day

        # 3. Only day is missing
        elif not np.isnat(date) and not day:
            day_number = (date.astype('datetime64[D]').view('int64') - 4) % 7
            day = day_names[day_number]
            row['day'] = day
            previous_day = day
        
        # 6. Both missing
        elif np.isnat(date) and not day:
            # Check if quarter same as previous quarter
            if row['quarter'] == data[i-1]['quarter']:
                row['date'] = previous_date
                day_number = (previous_date.astype('datetime64[D]').view('int64') - 4) % 7
                row['day'] = day_names[day_number]
            else: 
                # Check if the previous day was a Thursday and add two days since Friday is skipped.
                date = previous_date + np.timedelta64(1, 'D')
                if previous_day == "Thursday":
                    date += np.timedelta64(1, 'D')
                            
                    day_number = (date.astype('datetime64[D]').view('int64') - 4) % 7
                    day = day_names[day_number]
                    row['date'] = date
                    row['day'] = day
                    previous_date = date
                    previous_day = day
    
    return data


def process_quarter(data: np.ndarray) -> np.ndarray:
    """
    Processes the 'quarter' field of a numpy structured array.

    Fills the missing 'quarter' fields by using the day of the month from the 'date' field. 
    The month is divided into five quarters, each covering about a week. The days 1-7 belong to 'Quarter1', 
    8-14 belong to 'Quarter2', 15-21 to 'Quarter3', 22-28 to 'Quarter4', and 29-31 to 'Quarter5'.
    """

    for row in data:
        date_str = np.datetime_as_string(row['date'], unit='D')  # Convert to string
        date = datetime.strptime(date_str, '%Y-%m-%d')  # Convert to Python's datetime
        day  = date.day
        quarter = row['quarter']

        if not quarter:
            if 1 <= day <= 7:
                quarter = 'Quarter1'

            if 8 <= day <= 14:
                quarter = 'Quarter2'

            if 15 <= day <= 21:
                quarter = 'Quarter3'

            if 22 <= day <= 28:
                quarter = 'Quarter4'
            
            if 29 <= day <= 31:
                quarter = 'Quarter5'

            row['quarter'] = quarter

    return data   