import os 
import torch 
import mne 
from torch.utils.data import Dataset

class BaseEEGDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.transform = transform
        self.data_dir = data_dir
        self.data = None 
        self.labels = None

    def load_epochs(self, file_list, directory=None):
        epochs_list = []
        dir_path = directory or self.data_dir

        for file in file_list:
            file_path = os.path.join(dir_path, file)
            epochs = mne.read_epochs(file_path, preload=True, verbose=False)

            ch_names = epochs.info['ch_names']
            standard_ch_names = [f'EEG_{i+1:03d}' for i in range(len(ch_names))]
            ch_mapping = dict(zip(ch_names, standard_ch_names))
            epochs.rename_channels(ch_mapping)
            epochs_list.append(epochs)
        
        if not epochs_list:
            return None
        
        return mne.concatenate_epochs(epochs_list)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx].float()
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        if len(sample.shape) == 2:
            sample = sample.unsqueeze(0)

        return sample, label
    
class BCI2aDataset(BaseEEGDataset):
    def __init__(self, data_dir, mode='train', transform=None):
        if mode not in ['train', 'eval']:
            raise ValueError("Mode must be either 'train' or 'eval'")
        
        self.mode = mode
        
        super().__init__(data_dir, transform)
        
        self.foot_dir = os.path.join(data_dir, 'epochs_class_foot')
        self.left_dir = os.path.join(data_dir, 'epochs_class_left')
        self.right_dir = os.path.join(data_dir, 'epochs_class_right')
        self.tongue_dir = os.path.join(data_dir, 'epochs_class_tongue')

        foot_files = [f for f in os.listdir(self.foot_dir) if (mode == 'train' and 'T' in f) or (mode == 'eval' and 'E' in f)]
        left_files = [f for f in os.listdir(self.left_dir) if (mode == 'train' and 'T' in f) or (mode == 'eval' and 'E' in f)]
        right_files = [f for f in os.listdir(self.right_dir) if (mode == 'train' and 'T' in f) or (mode == 'eval' and 'E' in f)]
        tongue_files = [f for f in os.listdir(self.tongue_dir) if (mode == 'train' and 'T' in f) or (mode == 'eval' and 'E' in f)]

        foot_epochs = self.load_epochs(foot_files, self.foot_dir)
        left_epochs = self.load_epochs(left_files, self.left_dir)
        right_epochs = self.load_epochs(right_files, self.right_dir)
        tongue_epochs = self.load_epochs(tongue_files, self.tongue_dir)
        
        if any(epochs is None for epochs in [foot_epochs, left_epochs, right_epochs, tongue_epochs]):
            raise ValueError("One or more class directories are empty or invalid.")

        # Convert to tensors and concatenate
        self.data = torch.cat([
            torch.from_numpy(foot_epochs.get_data()).float(),
            torch.from_numpy(left_epochs.get_data()).float(),
            torch.from_numpy(right_epochs.get_data()).float(),
            torch.from_numpy(tongue_epochs.get_data()).float()
        ])
        
        # Assign labels (0: foot, 1: left hand, 2: right hand, 3: tongue)
        self.labels = torch.cat([
            torch.zeros(len(foot_epochs)),
            torch.ones(len(left_epochs)),
            torch.full((len(right_epochs),), 2),
            torch.full((len(tongue_epochs),), 3)
        ])
        self.labels = self.labels.long()

class CHBMITDataset(BaseEEGDataset):
    def __init__(self, data_dir, transform=None):
        super().__init__(data_dir, transform)

        # Setup directories and load file lists
        self.seizures_dir = os.path.join(data_dir, 'seizures')
        self.normals_dir = os.path.join(data_dir, 'normals')

        seizure_files = self._read_file_list(os.path.join(self.seizures_dir, "RECORDS_WITH_SEIZURES"))
        normal_files = self._read_file_list(os.path.join(self.normals_dir, "RECORDS_WITHOUT_SEIZURES"))
    
        # Load and combine data
        seizure_epochs = self.load_epochs(seizure_files, self.seizures_dir)
        normal_epochs = self.load_epochs(normal_files, self.normals_dir)

        self.data = torch.cat([
            torch.from_numpy(seizure_epochs.get_data()).float(),
            torch.from_numpy(normal_epochs.get_data()).float()
        ])
        self.labels = torch.cat([
            torch.ones(len(seizure_epochs)),
            torch.zeros(len(normal_epochs))
        ])
        self.labels = self.labels.float().unsqueeze(1)

    def _read_file_list(self, file_path):
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

class TUHDataset(BaseEEGDataset):
    def __init__(self, data_dir: str, transform=None):
        super().__init__(data_dir, transform)
        
        # Define subdirectory structure
        stage_dir = os.path.join(data_dir, 'edf_stage2_10_2', 'train')
        
        # Load file lists
        seizure_files = self._read_file_list(os.path.join(stage_dir, "RECORDS_WITH_SEIZURES.txt"))
        normal_files = self._read_file_list(os.path.join(stage_dir, "RECORDS_WITHOUT_SEIZURES.txt"))
        
        # Load and combine data
        seizure_epochs = self.load_epochs(seizure_files)
        normal_epochs = self.load_epochs(normal_files)
        
        self.data = torch.cat([
            torch.tensor(seizure_epochs.get_data()),
            torch.tensor(normal_epochs.get_data())
        ])
        self.labels = torch.cat([
            torch.ones(len(seizure_epochs.events)), 
            torch.zeros(len(normal_epochs.events))
        ])
    
    def _read_file_list(self, file_path):
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

class TUHDatasetMultiClass(BaseEEGDataset):
    def __init__(self, data_dir: str, transform=None):
        super().__init__(data_dir, transform)

        # Setup directories and load file lists
        stage_dir = os.path.join(data_dir, 'edf_stage2_multi_10_10', 'train')
        
        # Load file listst for different classes 
        class_files = {}
        self.class_names = ['bckg', 'absz', 'cpsz', 'fnsz', 'gnsz', 'mysz', 'spsz', 'tcsz', 'tnsz']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        for class_name in self.class_names:
            class_files[class_name] = self._compose_file_list(stage_dir, class_name)

        data_list = []
        labels_list = []

        for idx, (class_name, files) in enumerate(class_files.items()):
            if files: 
                epochs = self.load_epochs(files)
                data_list.append(torch.tensor(epochs.get_data()))

                class_idx = self.class_to_idx[class_name]
                class_labels = torch.full((len(epochs.events),), class_idx, dtype=torch.long)
                labels_list.append(class_labels)

        self.data = torch.cat(data_list) 
        self.labels = torch.cat(labels_list)

        print(f"Loaded dataset with {len(self.data)} samples across {len(self.class_names)} classes")
        # Print class distribution
        for class_name, idx in self.class_to_idx.items():
            count = (self.labels == idx).sum().item()
            print(f"Class {class_name}: {count} samples")

    def _compose_file_list(self, stage_dir, class_name):
        class_dir = os.path.join(stage_dir, class_name)
        return [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('epo.fif')]

def inspect_dataset(dataset, name="Dataset"):
    print(f"\nInspecting: {name}")
    print(f"Total samples: {len(dataset)}")

    sample, label = dataset[0]
    print(f"Sample shape: {sample.shape} (type: {type(sample)})")
    print(f"Label shape: {label.shape if hasattr(label, 'shape') else 'scalar'} (value: {label})")


if __name__ == "__main__":
    bci_dataset = BCI2aDataset(data_dir='../BCICIV_2a/stage1/', mode='train')
