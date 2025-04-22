from config import PARAMETERS
import numpy as np 
from mne.preprocessing import ICA
from autoreject import AutoReject

def get_parameters():
    n_parameters = 4
    mask = 0 
    parameters_set = []
    while mask < (1 << n_parameters):
        parameters = {}
        parameters['normalize'] = (mask >> 0) & 1
        parameters['filter'] = (mask >> 1) & 1
        window_size_index = ((mask >> 2) & 1) + ((mask >> 3) & 1) * 2
        if window_size_index < len(PARAMETERS[2]):
            parameters['window_size'] = PARAMETERS[2][window_size_index]

            for step_size in [0.25, 0.50, 1.0]:
                parameters['step_size'] = int(parameters['window_size'] * step_size)
                parameters_set.append(parameters.copy())
        mask += 1
    return parameters_set

def calculate_offset(start, end, timestamp_end, alpha=0.15, min_offset=5, max_offset=30):
    duration = end - start
    offset = duration * alpha
    offset = min(max_offset, max(min_offset, offset))
    return (max(0, start - offset), min(timestamp_end, end + offset))

def z_normalize(data):
    data = np.nan_to_num(data, nan=0.0)  # Handle NaN values
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    std[std == 0] = 1  # Avoid division by zero for flat signals
    return (data - mean) / std

def artifact_removal(epochs):
    ica = ICA(n_components=0.99, random_state=97)  # Use 99% of variance
    ica.fit(epochs.copy().filter(l_freq=1, h_freq=None))
        
    ica.exclude = [0, 1]
    ica.apply(epochs)

    n_folds = min(10, len(epochs))
    ar = AutoReject(n_interpolate=[1, 2, 3, 4], n_jobs=12, cv=n_folds)
    epochs_clean, log = ar.fit_transform(epochs, return_log=True)
    return epochs_clean