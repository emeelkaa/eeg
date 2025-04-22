import os
import mne
import sys
import pandas as pd
import random
import numpy as np 
import logging
from mne.channels import make_standard_montage, make_dig_montage
from config import SAMPLING_RATE, FILTER, NORMALIZE, WINDOW, STEP
from preprocessing.utils import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

ROOT_DIR = "../chbmit/stage1_dataset"
NEW_ROOT_DIR = None
NEW_ROOT_DIR_SZ = None
RECORDS_WITHOUT_SEIZURES = "RECORDS_WITHOUT_SEIZURES"

BIPLOAR_PAIRS = {
    "FP1-F7": ("Fp1", "F7"),
    "F7-T7": ("F7", "T7"),
    "T7-P7": ("T7", "P7"),
    "P7-O1": ("P7", "O1"),
    "FP1-F3": ("Fp1", "F3"),
    "F3-C3": ("F3", "C3"),
    "C3-P3": ("C3", "P3"),
    "P3-O1": ("P3", "O1"),
    "FZ-CZ": ("Fz", "Cz"),
    "CZ-PZ": ("Cz", "Pz"),
    "FP2-F4": ("Fp2", "F4"),
    "F4-C4": ("F4", "C4"),
    "C4-P4": ("C4", "P4"),
    "P4-O2": ("P4", "O2"),
    "FP2-F8": ("Fp2", "F8"),
    "F8-T8": ("F8", "T8"),
    "T8-P8": ("T8", "P8"),
    "P8-O2": ("P8", "O2"),
}

if __name__ == "__main__":    
    NEW_ROOT_DIR = os.path.join(f"stage2_dataset_{WINDOW}_{STEP}", "normals")
    
    metadata = {
        "subject_id": [],
        "file": [],
        "segment_start": [],
        "segment_end": [],
    }

    os.makedirs(NEW_ROOT_DIR, exist_ok=True)

    with open(os.path.join(ROOT_DIR, RECORDS_WITHOUT_SEIZURES), 'r') as f:
        records = f.read().splitlines()

    subjects = os.listdir(ROOT_DIR)
    subjects = [subject for subject in subjects if subject.startswith('chb')]
    subjects = sorted(subjects)

    chosen_records = set()

    for i in subjects:
        for j in range(4):
            patience = 0
            file_choice = random.choice([record for record in records if record.startswith(i)])
            while patience < 10 and file_choice in chosen_records:
                file_choice = random.choice([record for record in records if record.startswith(i)])
                patience += 1
            if patience == 10:
                continue

            chosen_records.add(file_choice)

            fif_file_path = os.path.join(ROOT_DIR, file_choice)

            logging.info(f'Processing file: {fif_file_path}')

            fif_file_name = file_choice.split('/')[1]
            subject_id = file_choice.split('/')[0]
            fif_file_name = fif_file_name.replace('_raw.fif', '_epo.fif')
            try:
                raw = mne.io.read_raw_fif(fif_file_path, preload=True, verbose=False)
                raw.resample(SAMPLING_RATE, npad='auto')
            except Exception as e:
                logging.error(f"Error loading {fif_file_path}: {e}")
                continue
            
            montage = make_standard_montage('standard_1020')
            ch_pos = montage.get_positions()['ch_pos']
            bipolar_locs = {
                channel: (ch_pos[ch1] + ch_pos[ch2]) / 2
                for channel, (ch1, ch2) in BIPLOAR_PAIRS.items()
            }
            montage = make_dig_montage(
                ch_pos=bipolar_locs,
                coord_frame='head'
            )
            raw.set_montage(montage)

            if FILTER:
                # filtering noise at 60 Hz, was confirmed by visual inspection, CHB-MIT dataset is collected in USA, so 60 Hz is the noise frequency
                raw.notch_filter(freqs=60, verbose=False)

                # filtering freq between 0.5 and 45 Hz, as it is common in EEG signal processing
                raw.filter(0.5, 45, fir_design='firwin', verbose=False)

            os.makedirs(os.path.join(NEW_ROOT_DIR, subject_id), exist_ok=True)

            duration = min(3600, raw.times[-1])
            seg_start = random.uniform(0, raw.times[-1] - duration)
            seg_end = seg_start + duration

            n_epochs = int((seg_end - seg_start - WINDOW) // STEP) + 1
        
            if n_epochs == 0:
                logging.warning(f"Segment {seg_start}-{seg_end} is too short for a {WINDOW}s epoch.")
                continue
            
            # Create event timings (in seconds)
            event_times = seg_start + np.arange(n_epochs) * STEP
            
            # Convert to sample numbers
            event_samples = (event_times * SAMPLING_RATE).astype(int)
            
            # Create event matrix: [sample, 0, event_id]
            events = np.column_stack([event_samples, np.zeros_like(event_samples), np.ones_like(event_samples)])

            all_events = events[events[:, 0].argsort()]
            # Create Epochs object
            epochs = mne.Epochs(raw, all_events, tmin=0, tmax=WINDOW - (1 / SAMPLING_RATE), baseline=None, preload=True, verbose=False)

            if NORMALIZE: 
                epochs.apply_function(z_normalize, picks='eeg')

            epochs.save(os.path.join(NEW_ROOT_DIR, subject_id, fif_file_name), overwrite=True)

            # saving to RECORDS_WITHOUT_SEIZURES file
            with open(os.path.join(NEW_ROOT_DIR, RECORDS_WITHOUT_SEIZURES), 'a') as file:
                file.write(f"{subject_id}/{fif_file_name}\n")

            # saving metadata    
            metadata['subject_id'].append(subject_id)
            metadata['file'].append(fif_file_name)
            metadata['segment_start'].append(seg_start)
            metadata['segment_end'].append(seg_end)
    
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(NEW_ROOT_DIR, "metadata.csv"), index=False)