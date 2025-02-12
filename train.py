import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import logging
from dataset import CHBMITDataset
from EEG_Conformer import Conformer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 5  # For early stopping
TRAIN_RATIO = 0.7  # (70/15/15)
VAL_RATIO = 0.15    

# Data loading
data_dir = "stage2_dataset_v2_4sec"
dataset = CHBMITDataset(data_dir)

# Split dataset into train, validation, and test sets
train_size = int(TRAIN_RATIO * len(dataset))
val_size = int(VAL_RATIO * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Use this in DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# Model, loss, optimizer
model = Conformer().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

def plot_confusion_matrix(y_true, y_pred, classes=None):
    cm = confusion_matrix(y_true, y_pred)
    if classes is None:
        classes = [f"Class {i}" for i in range(cm.shape[0])]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Calculate and log metrics
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    for i, class_name in enumerate(classes):
        logging.info(f"{class_name} - Precision: {precision[i]:.3f}, Recall: {recall[i]:.3f}, F1: {f1[i]:.3f}")

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device, non_blocking=True).float(), labels.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        
        _, outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    avg_loss = epoch_loss / len(dataloader)
    return avg_loss, accuracy

def validate_epoch(model, dataloader, criterion, device, get_predictions=False):
    model.eval()
    epoch_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device, non_blocking=True).float(), labels.to(device, non_blocking=True).long()

            _, outputs = model(inputs)

            loss = criterion(outputs, labels)

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if get_predictions:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    avg_loss = epoch_loss / len(dataloader)
    
    if get_predictions:
        return avg_loss, accuracy, all_preds, all_labels
    return avg_loss, accuracy

# Early stopping
best_loss = float('inf')
early_stop_counter = 0

# Training loop
for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    logging.info(f"Epoch {epoch+1}/{EPOCHS}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_acc:.2f}%")

    # Validation
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    logging.info(f"Epoch {epoch+1}/{EPOCHS}: Val Loss = {val_loss:.4f}, Val Accuracy = {val_acc:.2f}%")

    # Learning rate scheduling
    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), "best_eeg_conformer.pth")
        logging.info(f"New best model saved with val loss: {best_loss:.4f}")
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            logging.info("Early stopping triggered.")
            break

# Test the model on the test set and generate confusion matrix
model.load_state_dict(torch.load("best_eeg_conformer.pth"))
test_loss, test_acc, test_preds, test_labels = validate_epoch(model, test_loader, criterion, device, get_predictions=True)
logging.info(f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.2f}%")

# Get the number of classes from your dataset
num_classes = len(set(test_labels))
class_names = [f"Class {i}" for i in range(num_classes)]  # Replace with actual class names if available

# Plot and save confusion matrix
plot_confusion_matrix(test_labels, test_preds, classes=class_names)
logging.info("Confusion matrix saved as 'confusion_matrix.png'")

logging.info("Training and testing complete. Best model saved.")