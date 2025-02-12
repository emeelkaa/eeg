import os 
import torch 
import mne 
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import numpy as np

class CHBMITDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to the preprocessed dataset directory.
            transform (callable, optional)
        """
        self.seizures_dir = os.path.join(data_dir, 'seizures')
        self.normals_dir = os.path.join(data_dir, 'normals')
        self.transform = transform

        # Load file paths 
        seizure_file = os.path.join(self.seizures_dir, "RECORDS_WITH_SEIZURES")
        non_seizure_file = os.path.join(self.normals_dir, "RECORDS_WITHOUT_SEIZURES")

        with open(seizure_file, 'r') as f:
            self.seizure_files = [line.strip() for line in f.readlines()]
        
        with open(non_seizure_file, 'r') as f:
            self.non_seizure_files = [line.strip() for line in f.readlines()]

        # Load seizure data 
        self.seizure_epochs = self.load_epochs(self.seizure_files)
        num_seizure_epochs = len(self.seizure_epochs.events)

        # Load non-seizure data and balance it
        self.non_seizure_epochs = self.load_epochs(self.non_seizure_files, seizure=False)
        num_non_seizure_epochs = len(self.non_seizure_epochs.events)

        # Combine datasets
        self.data = torch.cat([torch.tensor(self.seizure_epochs.get_data()),
                               torch.tensor(self.non_seizure_epochs.get_data())])
        self.labels = torch.cat([torch.ones(num_seizure_epochs), torch.zeros(num_non_seizure_epochs)])

    def load_epochs(self, file_list, seizure=True):
        epochs_list = []

        dir = self.seizures_dir if seizure else self.normals_dir
        
        for file in file_list:
            file_path = os.path.join(dir, file)
            epochs = mne.read_epochs(file_path, preload=True, verbose=False)            
            epochs_list.append(epochs)

        return mne.concatenate_epochs(epochs_list)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx].float()
        label = self.labels[idx]

        sample = rearrange(sample, 'c t -> 1 c t')  # Add a dummy batch dimension

        if self.transform:
            sample = self.transform(sample)

        return sample, label
    
'''
dataset = CHBMITDataset('test/')
print(dataset.labels)
#dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

num_samples = len(dataset)
num_seizure = (dataset.labels == 1).sum().item()
num_non_seizure = (dataset.labels == 0).sum().item()

print(f"Total Samples: {num_samples}")
print(f"Seizure Samples: {num_seizure}")
print(f"Non-Seizure Samples: {num_non_seizure}")

'''
