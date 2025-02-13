import os
import mne
import numpy as np
import pandas as pd
import pickle
import random
import logging
from sklearn.neighbors import KernelDensity

logging.basicConfig(
    level=logging.INFO,  # Set the logging level (INFO, WARNING, ERROR, etc.)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format log messages
    datefmt="%Y-%m-%d %H:%M:%S",  # Timestamp format
)

# Define dataset directories
ROOT_DIR = "stage1_dataset"
NEW_ROOT_DIR = os.path.join("stage2_dataset_v2_4sec", "normals")
RECORDS_WITHOUT_SEIZURES = "RECORDS_WITHOUT_SEIZURES"

WINDOW_SIZE = 4
SAMPLING_RATE = 256 

current_subject_id = None

def get_durations_freq():
    """
    Retrieve seizure segment durations from metadata from KDE modeling
    """
    root_dir = os.path.join('stage2_dataset_v2','seizures')
    df = pd.read_csv(os.path.join(root_dir, "metadata.csv"))
    seizure_durations = df["segment_end"] - df["segment_start"]
    return seizure_durations.values.reshape(-1, 1)

def get_subjects_freq():
    """
    Get subjects' seizure occurrence frequencies for weighted sampling
    """
    root_dir = os.path.join('stage2_dataset_v2','seizures')
    df = pd.read_csv(os.path.join(root_dir, 'metadata.csv'))
    temp = df.groupby(df['subject_id']).count()
    seizure_counts = temp['file'].values
    weights = seizure_counts / np.sum(seizure_counts)  # normalize to probabilities
    sampled_ids = np.random.choice(df['subject_id'].unique(), size=df.shape[0], replace=True, p=weights)
    return sorted(sampled_ids)

def sample_duration(kde_model, min_dur=6.0, max_dur=752.0):
    while True:
        dur = kde_model.sample(1)[0][0]
        if min_dur <= dur <= max_dur:
            return dur

def z_normalize(data):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    return (data - mean) / std

def get_file_occurrences(file_choice):
    if not os.path.exists(os.path.join(NEW_ROOT_DIR, RECORDS_WITHOUT_SEIZURES)):  
        with open(os.path.join(NEW_ROOT_DIR, RECORDS_WITHOUT_SEIZURES), 'w') as f:
            pass
    with open(os.path.join(NEW_ROOT_DIR, RECORDS_WITHOUT_SEIZURES), 'r') as f:
        records = f.read().splitlines()
    return len([record for record in records if record.startswith(file_choice)])

if __name__ == "__main__":
    metadata = {
        "subject_id": [],
        "file": [],
        "segment_start": [],
        "segment_end": [],
    }

    if not os.path.exists(NEW_ROOT_DIR):
        os.makedirs(NEW_ROOT_DIR, exist_ok=True)
    else:
        counter = 0
        while os.path.exists(NEW_ROOT_DIR):
            NEW_ROOT_DIR = os.path.join("stage2_dataset_v2", f"normals{counter}")
            counter += 1

    kde_durations = KernelDensity(bandwidth=29.6150, kernel='gaussian')
    kde_durations.fit(get_durations_freq())

    with open(os.path.join(ROOT_DIR, RECORDS_WITHOUT_SEIZURES), 'r') as f:
        records = f.read().splitlines()

    subjects_freq = get_subjects_freq()
    for subject in subjects_freq:
        file_choice = random.choice([record for record in records if record.startswith(subject)])

        fif_file_path = os.path.join(ROOT_DIR, file_choice)

        logging.info(f'Processing file: {fif_file_path}')

        fif_file_name = file_choice.split('/')[1]
        subject_id = file_choice.split('/')[0]

        try:
            raw = mne.io.read_raw_fif(fif_file_path, preload=True, verbose=False)
        except Exception as e:
            logging.error(f"Error loading {fif_file_path}: {e}")
            continue

        # filtering noise at 60 Hz, was confirmed by visual inspection, CHB-MIT dataset is collected in USA, so 60 Hz is the noise frequency
        raw.notch_filter(freqs=60, verbose=False)

        # filtering freq between 0.5 and 45 Hz, as it is common in EEG signal processing
        raw.filter(0.5, 45, fir_design='firwin', verbose=False)

        if current_subject_id is None or current_subject_id != subject_id:
            current_subject_id = subject_id
            os.makedirs(os.path.join(NEW_ROOT_DIR, current_subject_id), exist_ok=True)

        # extracting the segment from raw data, and cropping it to the duration of a seizure
        duration = sample_duration(kde_durations)
        seg_start = random.uniform(0, raw.times[-1] - duration)
        seg_end = seg_start + duration
        start_idx, end_idx = raw.time_as_index([seg_start, seg_end])
        data, times = raw[:, start_idx:end_idx]

        # normalizing the data
        normalized_data = z_normalize(data)

        num_samples_per_epoch = WINDOW_SIZE * SAMPLING_RATE
        total_samples = normalized_data.shape[1]

        num_epochs = total_samples // num_samples_per_epoch

        if num_epochs == 0:
            logging.warning(f"Segment {seg_start}-{seg_end} is too short for a 10s epoch.")
            continue
            
        epochs_data = []
        events = []
        
        for i in range(num_epochs):
            epoch_start_idx = i * num_samples_per_epoch
            epoch_end_idx = (i + 1) * num_samples_per_epoch

            epoch_data = normalized_data[:, epoch_start_idx:epoch_end_idx]
            epochs_data.append(epoch_data)
            
            # Event marker for the epoch (using 1 for seizure)
            event_time = seg_start + (i * WINDOW_SIZE)
            events.append([i, 0, 0])  # i: sample index, 0: event onset, 1: normal event

        # Convert to MNE Epochs
        epochs_array = np.array(epochs_data)
        events_array = np.array(events)

        # Create MNE Epochs
        info = raw.info
        epochs = mne.EpochsArray(epochs_array, info, events=events_array, event_id={"normal": 0}, verbose=False)

        file_name = fif_file_name.replace('_raw.fif', f'_normal_{int(seg_start)}_{get_file_occurrences(file_choice)}_epo.fif')

        # saving new raw file
        epochs.save(os.path.join(NEW_ROOT_DIR, subject_id, file_name), overwrite=True)
                
        # saving to RECORDS_WITHOUT_SEIZURES file
        with open(os.path.join(NEW_ROOT_DIR, RECORDS_WITHOUT_SEIZURES), 'a') as file:
            file.write(f"{subject_id}/{file_name}\n")

        # saving metadata    
        metadata['subject_id'].append(subject_id)
        metadata['file'].append(file_name)
        metadata['segment_start'].append(seg_start)
        metadata['segment_end'].append(seg_end)

        logging.info(f"Saved {num_epochs} seizure epochs from {seg_start}s to {seg_end}s")
    
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(NEW_ROOT_DIR, "metadata.csv"), index=False)