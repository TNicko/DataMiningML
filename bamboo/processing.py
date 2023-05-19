# ----------------------------------------------------------------------------------------------------
# Description: Contains functions for processing different types of data in datasets.
# ----------------------------------------------------------------------------------------------------

import numpy as np
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

def read_csv(csv_file: str) -> np.ndarray:
    """Loads data from a CSV file into a numpy array. Infers data types directly from the data. """
    data = np.genfromtxt(csv_file, delimiter=',', dtype=None, names=True)
    return data


def convert_int_to_float(data: np.ndarray) -> np.ndarray:
    """
    Converts the integer columns in a structured numpy array to float.
    """
    new_dtypes = []
    for name, dtype in data.dtype.descr:
        if dtype == '<i8':
            new_dtypes.append((name, '<f8'))
        else:
            new_dtypes.append((name, dtype))
    
    return data.astype(new_dtypes)


def remove_columns(data: np.ndarray, columns: list[str]) -> np.ndarray:
    """    
    Removes specified columns from a numpy structured array and returns a new array.

    Parameters
    ----------
    data : A numpy structured array from which to remove columns.
    columns : A list of column names to remove from the data array.

    Returns
    -------
    np.ndarray : A new numpy array with the specified columns removed.
    """
    new_dtypes = dict(data.dtype.descr)
    for column in columns: del new_dtypes[column]

    # Initialize new structured array with updated data types (float instead of string)
    new_data = np.zeros(data.shape, dtype=list(new_dtypes.items()))
    
    # Add all columns from the original data and new categorical columns to the new array
    for column in new_data.dtype.names:
        new_data[column] = data[column]

    return new_data

def convert_dates(data: np.ndarray) -> np.ndarray:
    """Converts string or bytestring representations of dates into datetime objects in a numpy structured array."""
    # Step 1: Define a new dtype
    new_dtype = []
    for name, dtype in data.dtype.descr:
        if name == 'date':
            new_dtype.append((name, 'datetime64[D]'))  # change date to datetime
        else:
            new_dtype.append((name, dtype))
    
    # Step 2: Create a new array
    new_data = np.empty(data.shape, dtype=new_dtype)

    # Step 3: Copy values to new array
    for name in data.dtype.names:
        if name == 'date':
            # Convert date strings to datetime objects and store in new array
            new_data['date'] = [np.datetime64(datetime.strptime(date.decode('utf-8'), "%m/%d/%Y")) if date else 'NaT' for date in data['date']]
        else:
            new_data[name] = data[name]
    
    return new_data


def split_date_column(data: np.ndarray) -> np.ndarray:
    """Splits the 'date' field of a numpy structured array into three separate fields: 'year', 'month', and 'day_of_month'."""
    # Create new columns
    year = np.empty(data.shape[0], dtype=int)
    month = np.empty(data.shape[0], dtype=int)
    day_of_month = np.empty(data.shape[0], dtype=int)
    
    # Fill the new columns
    for i, row in enumerate(data):
        date = row['date']
        year[i] = np.datetime64(date, 'Y').astype(int) + 1970
        month[i] = np.datetime64(date, 'M').astype(int) % 12 + 1
        day_of_month[i] = (date - np.datetime64(date, 'M')).astype(int) + 1
    
    # Get dtype of the data array as dict
    data_dtypes = dict(data.dtype.descr)

    # Create new dtypes dict and add the old dtypes
    new_dtypes = {
        'year': int,
        'month': int,
        'day_of_month': int
    }
    new_dtypes.update(data_dtypes)
    del new_dtypes['date']

    # Create new structured array with the new dtypes with shape of the old array + 3
    new_data = np.empty(data.shape[0], dtype=list(new_dtypes.items()))

    # Copy the old data into the new array
    for col in data.dtype.names:
        if col != 'date':
            new_data[col] = data[col]
    
    # Copy new data to the new array
    new_data['year'] = year
    new_data['month'] = month
    new_data['day_of_month'] = day_of_month
    
    return new_data

def get_encoded_column(data: np.ndarray, column: str) -> np.ndarray:
    """Encodes the specified column of a numpy structured array using ordinal encoding."""
    # reshape column to 2D array
    data_column = data[:][column].reshape(-1, 1)
    
    # crete OrdinalEncoder and fit to column
    encoder = OrdinalEncoder(categories='auto', dtype=float)
    encoder.fit(data_column)

    # replace column with encoded values
    encoded_column = encoder.transform(data_column).flatten()

    return encoded_column

def encode_categorical_data(data: np.ndarray, categorical_cols: list[str]) -> np.ndarray:
    """Encodes the specified categorical columns of a numpy structured array using ordinal encoding."""
    new_dtypes = [(name, 'float64' if name in categorical_cols else dtype) for name, dtype in data.dtype.descr]
    category_columns = {}

    # Encode and store categorical columns
    for column in categorical_cols:
        encoded_column = get_encoded_column(data, column)
        category_columns[column] = encoded_column

    # Initialize new structured array with updated data types (float instead of string)
    encoded_data = np.zeros(data.shape, dtype=new_dtypes)
    
    # Add all columns from the original data and new categorical columns to the new array
    for column in data.dtype.names:
        if column in categorical_cols:
            encoded_data[column] = category_columns[column]
        else:
            encoded_data[column] = data[column]

    return encoded_data

def fill_missing_numbers(data: np.ndarray, column: np.ndarray) -> np.ndarray:
    """
    Fills the missing values in the specified column of a numpy structured array.
    It fills missing float values (np.nan) with the mean of the column.
    """
    imp = SimpleImputer(missing_values=np.nan, strategy='mean',)

    # fit and transform the data
    data_filled = imp.fit_transform(data[column].reshape(-1, 1))

    # reshape the data_filled array to have the same shape as the original column
    data_filled = data_filled.reshape(-1)

    # replace the original column with the new one
    data[column] = data_filled

    return data

def fill_categorical_missing(data: np.ndarray, column: str) -> np.ndarray:
    """
    Fills missing values in a given categorical column of a numpy structured array with 
    random values from the existing categories. The probability of a category being 
    selected is proportional to its occurrence frequency in the original data.

    Notes
    -----
    This function assumes that missing values are represented as empty byte strings (`b''`).
    """
    # Get unique categories and their counts, ignoring missing values
    unique, counts = np.unique(data[column], return_counts=True)

    # Remove the missing category
    mask = unique != b''
    unique = unique[mask]
    counts = counts[mask]

    # Calculate the probabilities for each category
    probabilities = counts / np.sum(counts)
    
    # Find the missing values in the column
    missing_mask = data[column] == b''
    
    # Generate a random array of categories, based on their weights, to fill missing values
    fill_values = np.random.choice(unique, size=np.sum(missing_mask), p=probabilities)
    
    # Fill the missing values
    data[column][missing_mask] = fill_values
    
    return data

def split_features_and_target(data: np.ndarray) -> tuple:
    """Splits the data into features and target arrays."""
    data_names = data.dtype.names
    X_names = data_names[:-1]
    y_name = data_names[-1]

    # Construct array of features
    X = np.empty((len(data), len(X_names)), dtype=float)
    for i, name in enumerate(X_names):
        X[:, i] = data[:][name].copy()

    y = data[:][y_name].copy()

    return X, y  

def get_feature_and_target_names(data: np.ndarray) -> list:
    """Returns a list of column names for the given data."""
    data_names = data.dtype.names
    X_names = data_names[:-1]
    y_name = data_names[-1]
    return X_names, y_name

def get_categorical_columns(data: np.ndarray) -> list:
    """Returns a list of the string columns that should be converted to categorical for the given data."""
    categorical_cols = [column for column, dtype in dict(data.dtype.fields).items() if dtype[0].char in {'S', 'U'}]
    return categorical_cols
