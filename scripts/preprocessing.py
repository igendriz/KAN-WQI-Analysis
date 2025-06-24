# === Libraries ===
import pandas as pd
import numpy as np
from scipy.signal import filtfilt, savgol_filter

# === Functions ===
def aggregate_df_by_time(df, datetime_col='created_date', resample_resolution='1min'):
    """
    Aggregate DataFrame by specified time resolution using mean aggregation.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing the time series data.
    datetime_col : str, optional
        Name of the datetime column. Default is 'created_date'.
    resample_resolution : str, optional
        Time resolution for resampling, e.g., '1min', '5min', '1H'. Default is '1min'.

    Returns:
    --------
    pd.DataFrame
        Aggregated DataFrame with mean values per time window.
    """

    # Convert datetime column to datetime format
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Set datetime column as index (if not already set)
    if df.index.name != datetime_col:
        df.set_index(datetime_col, inplace=True)

    # Floor to minutes to remove seconds/milliseconds (keeping it general as 'min')
    df.index = df.index.floor('min')

    # Aggregate data by specified resolution using mean
    df_resampled = df.resample(resample_resolution).mean().reset_index()

    return df_resampled

def impute_missing_values(df, cols_to_impute, order=2):
    """
    Fill missing values in selected columns using polynomial interpolation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing time series data.
    cols_to_impute : list
        List of column names to apply interpolation.
    order : int, optional
        Order of polynomial interpolation (default is 2).
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing values filled.
    """
    df_imputed = df.copy()  # Avoid modifying original data
    
    for col in cols_to_impute:
        # Apply polynomial interpolation for NaNs
        df_imputed[col] = df_imputed[col].interpolate(method='polynomial', order=order, limit_direction='both')

        # If some NaNs remain (e.g., at the edges), use forward/backward fill as a fallback
        df_imputed[col] = df_imputed[col].fillna(method='bfill').fillna(method='ffill')

    return df_imputed

def apply_filtfilt_smoothing(df, cols_to_smooth, window_size=5):
    """
    Apply a zero-phase moving average filter using filtfilt from SciPy.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing time series data.
    cols_to_smooth : list
        List of column names to apply smoothing.
    window_size : int
        The size of the moving average window (default is 5).
    
    Returns:
    --------
    pd.DataFrame
        A DataFrame with smoothed numerical columns.
    """
    df_smoothed = df.copy()  # Avoid modifying original data
    
    # Define the moving average filter kernel (h)
    h = np.ones(window_size) / window_size

    # Apply filtfilt smoothing to each column
    for col in cols_to_smooth:
        df_smoothed[col] = filtfilt(h, 1, df[col], padlen=0)  # padlen=0 avoids edge artifacts

    return df_smoothed

def apply_savgol_smoothing(df, cols_to_smooth, window_size=5, poly_order=2):
    """
    Apply Savitzky-Golay filter for polynomial-based smoothing.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing time series data.
    cols_to_smooth : list
        List of column names to apply smoothing.
    window_size : int
        The size of the moving window (must be odd, default is 5).
    poly_order : int
        Order of the polynomial fit (default is 2).

    Returns:
    --------
    pd.DataFrame
        A DataFrame with smoothed numerical columns.
    """
    df_smoothed = df.copy()  # Avoid modifying original data

    for col in cols_to_smooth:
        df_smoothed[col] = savgol_filter(df[col], window_length=window_size, polyorder=poly_order, mode='nearest')

    return df_smoothed

def apply_single_pole_filter(df, cols_to_smooth, alpha=0.1):
    """
    Apply a single-pole low-pass filter for exponential smoothing.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing time series data.
    cols_to_smooth : list
        List of column names to apply smoothing.
    alpha : float
        Smoothing factor (0 < alpha ≤ 1), closer to 1 means less smoothing.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with smoothed numerical columns.
    """
    df_smoothed = df.copy()  # Avoid modifying original data

    for col in cols_to_smooth:
        # Initialize filtered output array
        y = np.zeros_like(df[col].values)

        # First value remains the same (initialization)
        y[0] = df[col].values[0]

        # Apply recursive single-pole filtering
        for i in range(1, len(df[col])):
            y[i] = alpha * df[col].values[i] + (1 - alpha) * y[i-1]

        df_smoothed[col] = y  # Store the filtered signal

    return df_smoothed

def apply_single_pole_filtfilt(df, cols_to_smooth, alpha=0.1):
    """
    Apply a zero-phase (non-recursive) single-pole low-pass filter using filtfilt.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with time series data.

    cols_to_smooth : list
        List of column names to apply zero-phase smoothing.

    alpha : float
        Smoothing factor (0 < alpha ≤ 1). Controls the cutoff behavior.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with smoothed values using filtfilt (zero-phase response).
    """

    # Define IIR filter coefficients for single-pole low-pass filter
    b = [alpha]                # numerator coefficients
    a = [1, -(1 - alpha)]      # denominator coefficients (recursive)

    df_smoothed = df.copy()

    for col in cols_to_smooth:
        x = df[col].values
        # Apply zero-phase filtering (forward and backward)
        y = filtfilt(b, a, x, method='pad')
        df_smoothed[col] = y

    return df_smoothed

def compute_quality_ratings(water_pH, TDS, water_temp):
    # Compute quality ratings (q_i)
    q_pH = ((water_pH - 7) / (8.5 - 7)) * 100
    q_TDS = (1 - (TDS / 500)) * 100
    q_Temp = (1 - abs(water_temp - 26) / 3) * 100
    return q_pH, q_TDS, q_Temp
    

def compute_wqi_from_values(water_pH, TDS, water_temp):
    """
    Compute Water Quality Index (WQI) from individual water parameter values.

    Parameters:
    -----------
    water_pH : float
        Measured pH value.
    TDS : float
        Measured Total Dissolved Solids (mg/L).
    water_temp : float
        Measured water temperature (°C).

    Returns:
    --------
    float
        Computed WQI value (0 to 100).
    """
    
    # Compute quality ratings (q_i)
    q_pH, q_TDS, q_Temp = compute_quality_ratings(water_pH, TDS, water_temp)

    # Define weights (inverse of permissible values)
    w_pH = 1 / 8.5
    w_TDS = 1 / 500
    w_Temp = 1 / 3

    Ws, qs = np.array([w_pH, w_TDS, w_Temp]), np.array([q_pH, q_TDS, q_Temp])
    # Compute WQI
    # WQI = (w_pH * q_pH + w_TDS * q_TDS + w_Temp * q_Temp) / (w_pH + w_TDS + w_Temp)

    WQI = np.dot(Ws,qs)/np.sum(Ws)

    return WQI

def compute_wqm_from_values(water_pH, TDS, water_temp):
    """
    Compute Water Quality Index (WQI) using a weighted root-mean-square formulation.

    Parameters:
    -----------
    water_pH : float
        Measured pH value.
    TDS : float
        Measured Total Dissolved Solids (mg/L).
    water_temp : float
        Measured water temperature (°C).

    Returns:
    --------
    float
        Computed WQI value (0 to 100 scale).
    """
    # Compute individual quality scores
    q_pH, q_TDS, q_Temp = compute_quality_ratings(water_pH, TDS, water_temp)

    # Define weights (e.g., inverse of permissible or standard limits)
    w_pH = 1 / 8.5
    w_TDS = 1 / 500
    w_Temp = 1 / 3

    # Stack into shape (N, 3)
    qs = np.column_stack([q_pH, q_TDS, q_Temp])
    Ws = np.array([w_pH, w_TDS, w_Temp])

    # Weighted Quadratic Mean - WQI
    WQI = np.sqrt(np.sum(qs**2 * Ws, axis=1))

    return WQI

def compute_wqiRMS_from_values(water_pH, TDS, water_temp):
    """
    Compute Water Quality Index (WQI) using a weighted root-mean-square formulation.

    Parameters:
    -----------
    water_pH : float
        Measured pH value.
    TDS : float
        Measured Total Dissolved Solids (mg/L).
    water_temp : float
        Measured water temperature (°C).

    Returns:
    --------
    float
        Computed WQI value (0 to 100 scale).
    """
    # Compute individual quality scores
    q_pH, q_TDS, q_Temp = compute_quality_ratings(water_pH, TDS, water_temp)

    # Define weights (e.g., inverse of permissible or standard limits)
    w_pH = 1 / 8.5
    w_TDS = 1 / 500
    w_Temp = 1 / 3

    # Stack into shape (N, 3)
    qs = np.column_stack([q_pH, q_TDS, q_Temp])
    Ws = np.array([w_pH, w_TDS, w_Temp])

    # Weighted RMS calculation
    # Compute weighted RMS WQI from quality scores (qs: shape [N, 3])
    WQI_RMS = np.sqrt(np.mean(qs**2 *Ws, axis=1))

    return WQI_RMS

def select_time_slice_df(df, cols=None, start_date=None, end_date=None):
    """
    Select a time-based slice from a DataFrame using the 'created_date' column and optional column filtering.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame. Must contain a 'created_date' column of datetime type.

    cols : list of str or None, optional
        List of columns to retain in addition to 'created_date'.
        If None, all columns are retained.
        Any column not found in the DataFrame will be ignored silently (except 'created_date').

    start_date : str or pd.Timestamp or None, optional
        Start of the time window. If None, selection starts from the earliest timestamp.

    end_date : str or pd.Timestamp or None, optional
        End of the time window. If None, selection continues to the latest timestamp.

    Returns:
    --------
    df_selected : pd.DataFrame
        A new DataFrame sliced by time and filtered by columns, always retaining 'created_date'.

    Raises:
    -------
    ValueError:
        If 'created_date' is not in the DataFrame or if start_date > end_date.

    Example:
    --------
    df_selected = select_time_slice_df(
        df_agg_min,
        cols=['temperature', 'TDS'],
        start_date='2023-01-30 12:00:00',
        end_date='2023-02-01 00:00:00'
    )
    """
    import pandas as pd

    # Ensure 'created_date' column exists
    if 'created_date' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'created_date' column.")

    # Convert dates to Timestamps if needed
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
    else:
        start_date = df['created_date'].min()

    if end_date is not None:
        end_date = pd.to_datetime(end_date)
    else:
        end_date = df['created_date'].max()

    # Check date validity
    if start_date > end_date:
        raise ValueError("start_date must be earlier than or equal to end_date.")

    # Filter by time window
    mask = (df['created_date'] >= start_date) & (df['created_date'] <= end_date)
    df_filtered = df.loc[mask].copy().reset_index(drop=True)

    # Determine final columns
    if cols is None:
        selected_columns = df_filtered.columns.tolist()
    else:
        selected_columns = ['created_date'] + [col for col in cols if col in df_filtered.columns and col != 'created_date']

    return df_filtered[selected_columns]
