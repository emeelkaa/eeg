import os
import mne
import pandas as pd
import pickle
from datetime import timedelta

ROOT_DIR = "stage0_dataset"
NEW_ROOT_DIR = "stage1_dataset"
RECORDS = "RECORDS"
DOUBLE_BANANA = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FZ-CZ', 'CZ-PZ', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2']


current_subject_id = None
current_labels = None
current_summary = None

old_style_summary = None


def save_current():
    if current_subject_id == None:
        return
    
    df = pd.DataFrame(current_summary)
    df.to_csv(os.path.join(NEW_ROOT_DIR, current_subject_id, f"{current_subject_id}_summary.csv"), index=False)

    with open(os.path.join(NEW_ROOT_DIR, current_subject_id, f"{current_subject_id}_labels.pkl"), 'wb') as file:
        pickle.dump(current_labels, file)

def extract_data(edf_file_name, start_time = None, duration = None):
    if current_subject_id == None:
        return
    
    data = {'start_time': None, 'end_time': None, 'num_seizures': 0, 'seizures': []}

    found = False
    current_index = 0
    while current_index < len(old_style_summary):
        if old_style_summary[current_index] == f"File Name: {edf_file_name}":
            found = True
            break
        current_index += 1
    
    if current_subject_id != 'chb24':
        current_index += 1
        data['start_time'] = old_style_summary[current_index].split()[-1]
        current_index += 1
        data['end_time'] = old_style_summary[current_index].split()[-1]
    else:
        data['start_time'] = (start_time).strftime('%H:%M:%S')  
        data['end_time'] = (start_time + timedelta(seconds=duration)).strftime('%H:%M:%S')
        if not found:
            return data

    current_index += 1
    data['num_seizures'] = int(old_style_summary[current_index].split()[-1])

    for i in range(data['num_seizures']):
        current_index += 1
        start_time = old_style_summary[current_index].split()[-2]
        current_index += 1
        end_time = old_style_summary[current_index].split()[-2]
        data['seizures'].append((int(start_time), int(end_time)))
    
    return data

    
if __name__ == "__main__":
    os.makedirs(NEW_ROOT_DIR, exist_ok=True)

    with open(os.path.join(ROOT_DIR, RECORDS), 'r') as file:
        records = file.read().splitlines()


    for record in records:
        edf_file_path = os.path.join(ROOT_DIR, record)

        edf_file_name = record.split('/')[1]
        subject_id = record.split('/')[0]

        raw = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False, exclude=['-', '.'])

        # removing duplicated channels as CHB-MIT, as inspected only 'T8-P8' channels is duplicated and duplicants are identical
        if 'T8-P8-0' in raw.ch_names and 'T8-P8-1' in raw.ch_names:
            raw.drop_channels(['T8-P8-1'])
            raw.rename_channels(mapping={'T8-P8-0': 'T8-P8'})

        # if montage is not double banana, skip it
        if not set(DOUBLE_BANANA).issubset(set(raw.ch_names)):
            continue
        
        # if montage has redundant channels, pick only double banana
        raw.pick(DOUBLE_BANANA)

        # changing the year of the date to 2023, as they put the year 2076 in the dataset 
        raw.set_meas_date(raw.info['meas_date'].replace(year=2023))

        if current_subject_id == None or subject_id != current_subject_id:
            save_current()
            
            current_subject_id = subject_id
            os.makedirs(os.path.join(NEW_ROOT_DIR, current_subject_id), exist_ok=True)
            current_labels = {}
            current_summary = {'file_name': [], 'start_time': [], 'end_time': [], 'num_seizures': []}

            with open(os.path.join(ROOT_DIR, current_subject_id, f"{current_subject_id}-summary.txt"), 'r') as file:
                old_style_summary = file.read().splitlines()

        data = extract_data(
            edf_file_name, 
            raw.info['meas_date'] if subject_id == 'chb24' else None, 
            raw.times[-1] if subject_id == 'chb24' else None
        )
        
        current_labels[edf_file_name.replace('.edf', '')] = data['seizures']
        current_summary['file_name'].append(edf_file_name.replace('.edf', ''))
        current_summary['start_time'].append(data['start_time'])
        current_summary['end_time'].append(data['end_time'])
        current_summary['num_seizures'].append(data['num_seizures'])

        with open(os.path.join(NEW_ROOT_DIR, 'RECORDS'), 'a') as file:
            file.write(f"{subject_id}/{edf_file_name.replace('.edf', '_raw.fif')}\n")

        if data['num_seizures'] > 0:
            with open(os.path.join(NEW_ROOT_DIR, 'RECORDS_WITH_SEIZURES'), 'a') as file:
                file.write(f"{subject_id}/{edf_file_name.replace('.edf', '_raw.fif')}\n")
        else:
            with open(os.path.join(NEW_ROOT_DIR, 'RECORDS_WITHOUT_SEIZURES'), 'a') as file:
                file.write(f"{subject_id}/{edf_file_name.replace('.edf', '_raw.fif')}\n")

        raw.save(os.path.join(NEW_ROOT_DIR, current_subject_id, f"{edf_file_name.replace('.edf', '_raw.fif')}"), overwrite=True)

    save_current()

    subject_info = []
    with open(os.path.join(ROOT_DIR, 'SUBJECT-INFO'), 'r') as file:
        subject_info = file.read().splitlines()

    subject_info = [subject.split('\t') for subject in subject_info]

    subject_info = {
        subject_info[0][0]: [subject_info[i][0] for i in range(1, len(subject_info))],
        subject_info[0][1]: [subject_info[i][1] for i in range(1, len(subject_info))],
        subject_info[0][2]: [subject_info[i][2] for i in range(1, len(subject_info))],
    }
    
    df = pd.DataFrame(subject_info)
    df.to_csv(os.path.join(NEW_ROOT_DIR, 'SUBJECT-INFO.csv'), index=False)
    