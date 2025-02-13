import os
import mne
import pickle
import numpy as np
import shutil
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,  # Set the logging level (INFO, WARNING, ERROR, etc.)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format log messages
    datefmt="%Y-%m-%d %H:%M:%S",  # Timestamp format
)

# Define dataset directories
ROOT_DIR = "stage1_dataset"
NEW_ROOT_DIR = os.path.join("stage2_dataset_v2_4sec", "seizures")
RECORDS_WITH_SEIZURES = "RECORDS_WITH_SEIZURES"

WINDOW_SIZE = 4
SAMPLING_RATE = 256 

current_subject_id = None
current_labels = None

def calculate_offset(start, end, timestamp_end, alpha=0.15, min_offset=5, max_offset=30):
    """
    Calculates the offset around the seizure event.
    Ensures that the segment is neither too shor nor too long
    """
    duration = end - start
    offset = duration * alpha   
    offset = min(max_offset, max(min_offset, offset))
    return (max(0, start - offset), min(timestamp_end, end + offset))

def z_normalize(data):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    return (data - mean) / std

def create_epochs_from_segment(data, sfreq, window_size):
    window_samples = int(window_size * sfreq)
    n_channels, n_samples = data.shape

    n_windows = n_samples // window_samples

    epochs = []

    for i in range(n_windows):
        start_idx = i * window_samples
        end_idx = (i + 1) * window_samples
        epoch = data[:, start_idx:end_idx]
        epochs.append(epoch)
    
    return epochs

if __name__ == "__main__":
    metadata = {
        "subject_id": [],
        "file": [],
        "seizure_start": [],
        "seizure_end": [],
        "segment_start": [],
        "segment_end": [],
    } 
    os.makedirs(NEW_ROOT_DIR, exist_ok=True)
    print("Started preprocessing...")
    # Read the list of seizure records
    with open(os.path.join(ROOT_DIR, RECORDS_WITH_SEIZURES), 'r') as f:
        records = f.read().splitlines()

    for record in records:
        fif_file_path = os.path.join(ROOT_DIR, record)
        logging.info(f'Processing file: {fif_file_path}')

        fif_file_name = record.split('/')[1]
        subject_id = record.split('/')[0]

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

            with open(os.path.join(ROOT_DIR, current_subject_id, f"{current_subject_id}_labels.pkl"), 'rb') as f:
                current_labels = pickle.load(f)

        file_key = fif_file_name.replace('_raw.fif', '')

        if  file_key in current_labels and len(current_labels[file_key]) > 0: # it is file with seizures
            seizure_times = current_labels[file_key]
            seizure_counter = 0

            for start, end in seizure_times:
                # Calculate segment boundaries with offset
                seg_start, seg_end = calculate_offset(int(start), int(end), raw.times[-1])
                
                # Extract segment data from raw EEG
                start_idx, end_idx = raw.time_as_index([seg_start, seg_end])
                data, times = raw[:, start_idx:end_idx]

                # Normalize EEG data
                normalized_data = z_normalize(data)

                # Split into 10-second epochs
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
                    events.append([i, 0, 1])  # i: sample index, 0: event onset, 1: seizure event

                # Convert to MNE Epochs
                epochs_array = np.array(epochs_data)
                events_array = np.array(events)

                # Create MNE Epochs
                info = raw.info
                epochs = mne.EpochsArray(epochs_array, info, events=events_array, event_id={"seizure": 1}, verbose=False)

                # Save the epochs
                file_name = fif_file_name.replace('_raw.fif', f'_seizure{seizure_counter}_epo.fif')
                epochs.save(os.path.join(NEW_ROOT_DIR, subject_id, file_name), overwrite=True)

                # Append to RECORDS_WITH_SEIZURES
                with open(os.path.join(NEW_ROOT_DIR, RECORDS_WITH_SEIZURES), 'a') as file:
                    file.write(f"{subject_id}/{file_name}\n")
                # Store metadata for the segment   
                metadata['subject_id'].append(subject_id)
                metadata['file'].append(file_name)
                metadata['seizure_start'].append(int(start))
                metadata['seizure_end'].append(int(end))
                metadata['segment_start'].append(seg_start)
                metadata['segment_end'].append(seg_end)

                logging.info(f"Saved {num_epochs} seizure epochs from {seg_start}s to {seg_end}s")
                seizure_counter += 1
        else: 
            raise KeyError("Ты пиздец охуел датасет")
     
        logging.info(f'Finish processing file: {fif_file_path}')
        
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(NEW_ROOT_DIR, "metadata.csv"), index=False)
