# scipy.signal.envelope 
import sys, importlib, os
import McsPy.McsData
import McsPy.McsCMOS
from McsPy import ureg, Q_

# matplotlib.pyplot will be used in these examples to generate the plots visualizing the data
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
import numpy as np
from scipy import signal

# import datashader as ds
# import datashader.transfer_functions as tf
import pandas as pd
import numpy as np
# import pynapple as nap


import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

na = np.array
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import h5py


import numpy as np
import scipy.signal

def detect_extreme_events(signal, 
                          chunk_size=5000, 
                          window_size=5000,
                          threshold_std=5, 
                          min_separation=2000, 
                          min_event_length=500, 
                          dt=0.0004):
    """
    Detects extreme low events in a signal using local (per-chunk) thresholds.
    Applies Hilbert transform to extract peaks, ensuring minimum separation and minimum event length.

    Parameters:
    signal (np.array): 1D array of signal data.
    chunk_size (int): Size of chunks for computing local thresholds.
    window_size (int): Number of samples to extract around each detected event peak.
    threshold_std (float): Threshold in terms of standard deviations (applied locally).
    min_separation (int): Minimum number of points between events.
    min_event_length (int): Minimum length of a detected event.
    dt (float): Time step for converting indices to time.

    Returns:
    list of np.array: Extracted event signal windows.
    list of list: Time intervals for the events in seconds.
    """
    windows = []
    intervals = []

    # Compute Hilbert transform and envelope once for entire signal
    analytic_signal = scipy.signal.hilbert(signal)
    envelope = np.abs(analytic_signal)
    baselines = []
    used_ranges = []  # to store [start, end) of accepted windows

    num_chunks = int(np.ceil(len(signal) / chunk_size))
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(signal))
        chunk = envelope[start_idx:end_idx]
        
        if len(chunk) < min_event_length:
            continue  # Skip small chunks
        
        # Local stats
        mean_val = np.mean(chunk)
        std_val = np.std(chunk)
        upper_threshold = mean_val + (threshold_std * std_val)
        end_threshold = mean_val + (.5* std_val)
        baselines.append([[start_idx,end_idx],[threshold_std*std_val,threshold_std*std_val]])
        
        # Local extreme low indices
        local_indices = np.where(chunk > upper_threshold)[0] + start_idx

        # Filter by minimum separation
        filtered_indices = []
        last_idx = -min_separation
        for idx in local_indices:
            if idx - last_idx >= min_separation:
                filtered_indices.append(idx)
                last_idx = idx

        # Extract windows centered at local peak of the envelope
        for idx in filtered_indices:
            peak_idx = idx + np.argmax(envelope[idx:min(idx + window_size, len(signal))])
            # window_start = max(0, peak_idx - window_size // 2)
            # window_end = min(len(signal), peak_idx + window_size // 2)

            end_idx = peak_idx
            while end_idx < len(signal) and envelope[end_idx] > end_threshold:
                end_idx += 1
            window_start = max(0, peak_idx - window_size // 2)
            window_end =min(len(signal), end_idx)
            
            if (window_end - window_start) < min_event_length:
                continue

            # Check for overlap with previously accepted windows
            overlap = any(
                not (window_end <= existing_start or window_start >= existing_end)
                for existing_start, existing_end in used_ranges
            )
            if overlap:
                continue

            # Accept the window
            windows.append(signal[window_start:window_end])
            intervals.append([float(window_start) * dt, float(window_end) * dt])
            used_ranges.append((window_start, window_end))

    return windows, intervals,baselines