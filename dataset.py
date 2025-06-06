import os 
import torch 
import mne 
from torch.utils.data import Dataset
from collections import Counter
import numpy as np 

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
            if not os.path.exists(file_path):
                continue  # Skip this file if it doesn't exist
            
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

def inspect_dataset(dataset, name="Dataset"):
    print(f"\n{'='*50}")
    print(f"Inspecting: {name}")
    print(f"{'='*50}")
    
    # Basic information
    print(f"Total samples: {len(dataset)}")
    
    # Data shape information
    print("\nData Information:")
    sample, label = dataset[0]
    print(f"  Sample shape: {sample.shape} (type: {type(sample).__name__})")
    print(f"  Label type: {type(label).__name__} (value: {label})")

    # Analyze label distribution
    print("\nLabel Distribution:")
    labels = [dataset[i][1] for i in range(len(dataset))]
    
    # For numeric labels
    if isinstance(label, (int, float, np.integer, np.floating)) or (hasattr(label, 'item') and isinstance(label.item(), (int, float))):
        # Convert tensor to number if needed
        if hasattr(label, 'item'):
            labels = [l.item() if hasattr(l, 'item') else l for l in labels]
            
        # Count unique labels
        unique_labels = set(labels)
        for lbl in sorted(unique_labels):
            count = labels.count(lbl)
            percentage = (count / len(dataset)) * 100
            print(f"  Class {lbl}: {count} samples ({percentage:.2f}%)")
            
        # Check if balanced
        counts = [labels.count(lbl) for lbl in unique_labels]
        imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
        print(f"  Imbalance ratio (max/min): {imbalance_ratio:.2f}")
    
    # For one-hot encoded or multi-label
    elif isinstance(label, (list, np.ndarray)) or (hasattr(label, 'numpy') and label.dim() > 0):
        if hasattr(label, 'numpy'):
            # For PyTorch tensors
            try:
                binary_check = all(l.item() in [0, 1] for l in label.flatten()) if label.numel() > 0 else False
            except:
                binary_check = False
        else:
            # For numpy arrays or lists
            flat_label = np.array(label).flatten()
            binary_check = all(l in [0, 1] for l in flat_label) if len(flat_label) > 0 else False
            
        if binary_check:
            print("  Multi-label or one-hot encoded format detected")
            
            # Get the number of classes from the first sample's label shape
            if hasattr(label, 'shape'):
                num_classes = label.shape[0]
            else:
                num_classes = len(label)
                
            # Count for each class
            for class_idx in range(num_classes):
                if hasattr(labels[0], 'numpy'):
                    # For PyTorch tensors
                    class_count = sum(l[class_idx].item() for l in labels)
                else:
                    # For numpy arrays or lists
                    class_count = sum(l[class_idx] for l in labels)
                
                percentage = (class_count / len(dataset)) * 100
                print(f"  Class {class_idx}: {class_count} samples ({percentage:.2f}%)")
        else:
            print("  Complex label format detected, distribution analysis skipped")
    else:
        print("  Non-standard label format, distribution analysis skipped")

    print(f"{'='*50}\n")


if __name__ == "__main__":
    dataset = CHBMITDataset(data_dir='../chbmit/stage2_dataset_ablation/noFilter_noNorm_4_1')
    inspect_dataset(dataset)