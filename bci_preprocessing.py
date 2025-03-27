import mne
import os
import sys
import numpy as np
import scipy.io
from mne.preprocessing import EOGRegression
from mne.preprocessing import ICA
from autoreject import AutoReject

CHANNEL_MAPPING = {
    'EEG-Fz': 'Fz',
    'EEG-0': 'FC3',
    'EEG-1': 'FC1',
    'EEG-2': 'FCz',
    'EEG-3': 'FC2',
    'EEG-4': 'FC4',
    'EEG-5': 'C5',
    'EEG-C3': 'C3',
    'EEG-6': 'C1',
    'EEG-Cz': 'Cz',
    'EEG-7': 'C2',
    'EEG-C4': 'C4',
    'EEG-8': 'C6',
    'EEG-9': 'CP3',
    'EEG-10': 'CP1',
    'EEG-11': 'CPz',
    'EEG-12': 'CP2',
    'EEG-13': 'CP4',
    'EEG-14': 'P1',
    'EEG-Pz': 'Pz',
    'EEG-15': 'P2',
    'EEG-16': 'POz',
}

CLASSES = [
    '769', #0x0301 Cue onset left (class 1)
    '770', #0x0302 Cue onset right (class 2)
    '771', #0x0303 Cue onset foot (class 3)
    '772', #0x0304 Cue onset tongue (class 4)
    '783', #0x030F Cue unknown
]

OUTPUT_DIRS = {
    1: 'epochs_class_left',     # Class 1 (769 - left hand)
    2: 'epochs_class_right',    # Class 2 (770 - right hand)
    3: 'epochs_class_foot',     # Class 3 (771 - foot)
    4: 'epochs_class_tongue'    # Class 4 (772 - tongue)
}

ROOT_DIR = 'stage0'
TARGET_DIR = 'stage1'
TRUE_LABESL_DIR = 'true_labels'

def z_normalize(data):
    data = np.nan_to_num(data, nan=0.0)  # Handle NaN values
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    std[std == 0] = 1  # Avoid division by zero for flat signals
    return (data - mean) / std

def autoreject(epochs):
    if len(sys.argv) > 1 and 'ar' in sys.argv:
        n_folds = min(10, len(epochs))
        ar = AutoReject(n_interpolate=[1, 2, 3, 4], n_jobs=12, cv=n_folds)
        epochs_clean, log = ar.fit_transform(epochs, return_log=True)
        return epochs_clean, log
    else:
        return epochs

def eog(raw):
    if len(sys.argv) > 1 and 'eog' in sys.argv:
        weights = weights = EOGRegression().fit(raw)
        raw_clean = weights.apply(raw, copy=True)
        return raw_clean
    return raw

if __name__ == '__main__':
    sessions = os.listdir(ROOT_DIR)

    for dir_path in OUTPUT_DIRS.values():
        os.makedirs(os.path.join(TARGET_DIR, dir_path), exist_ok=True)

    for session in sessions:
        raw = mne.io.read_raw_gdf(os.path.join(ROOT_DIR, session), preload=True)
        raw.set_channel_types({'EOG-central': 'eog', 'EOG-left': 'eog', 'EOG-right': 'eog'}, verbose=False)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.rename_channels(CHANNEL_MAPPING, verbose=False)
        raw.set_montage(montage, verbose=False)
        raw.set_eeg_reference("average", projection=True, verbose=False)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        raw = eog(raw)
        raw.drop_channels(['EOG-central', 'EOG-left', 'EOG-right'])

        filtered_event_ids = [event_id[key] for key in event_id.keys() if key in CLASSES]

        epochs = mne.Epochs(
            raw,
            events,
            event_id=filtered_event_ids,
            tmin=0,
            tmax=1.252,
            baseline=(0, 0),
            preload=True,
            verbose=False
        )
        epochs.apply_function(z_normalize, picks='eeg')

        reject_log = None
        if len(sys.argv) > 1 and 'ar' in sys.argv:
            epochs, reject_log = autoreject(epochs)

        mat = scipy.io.loadmat(os.path.join(TRUE_LABESL_DIR, session.replace('.gdf', '.mat')))
        classlabels = np.array([label[0] for label in mat['classlabel']], dtype=int)

        if reject_log is not None:
            classlabels = np.array(classlabels)[~reject_log.bad_epochs]

        if len(epochs) != len(classlabels):
            raise RuntimeError(f"Mismatch in {session}: {len(epochs)} epochs vs {len(classlabels)} labels")
            
        for class_num in [1, 2, 3, 4]:
            class_mask = np.array(classlabels) == class_num
            
            if not np.any(class_mask):
                raise RuntimeError(f"No epochs found for class {class_num} in {session}")
            
            class_epochs = epochs[class_mask]
            
            output_path = os.path.join(TARGET_DIR, OUTPUT_DIRS[class_num], f"{session.replace('.gdf', '')}_class{class_num}-epo.fif")
            class_epochs.save(output_path, overwrite=True)
            