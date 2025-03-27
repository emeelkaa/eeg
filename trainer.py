import logging
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# Optional
#from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
#from torch.utils.data import Subset

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, test_dataset, 
                 num_classes=2, batch_size=16, learning_rate=5e-4, 
                 weight_decay=1e-4, device=None, save_dir="./results"):
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

        # Configure data loaders with efficient settings
        loader_kwargs = {'batch_size': batch_size, 
                         'pin_memory': True,
                         'num_workers': 4}
        
        self.train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        self.val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        self.test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

        self.model = model.to(self.device)

        self.criterion = (
            nn.BCEWithLogitsLoss() if num_classes == 2
            else nn.CrossEntropyLoss()
        )

        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode="min", 
            patience=3, 
            factor=0.5
        )

        self.history = {
            'train_loss': [], 
            'train_acc': [], 
            'val_loss': [], 
            'val_acc': []
        }

        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _step(self, batch, training=True):
        inputs, labels = [x.to(self.device, non_blocking=True) for x in batch]

        if training:
            self.optimizer.zero_grad(set_to_none=True)
        
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        if training:
            loss.backward()
            self.optimizer.step()

        # Handle prediction differently for binary and multi-class
        preds = (
            (torch.sigmoid(outputs) > 0.5).float() if self.num_classes == 2
            else torch.argmax(outputs, dim=1)
        )

        correct = (preds == labels).sum().item()

        return loss.item(), correct, len(labels), preds, labels

    def _evaluate(self, dataloader, training=False):
        self.model.train() if training else self.model.eval()

        total_loss, total_correct, total_samples = 0, 0, 0
        preds_list, labels_list = [], []

        with torch.set_grad_enabled(training):
            for batch in tqdm(dataloader, desc="Training" if training else "Evaluation"):
                loss, correct, samples, preds, labels = self._step(batch, training)

                total_loss += loss
                total_correct += correct
                total_samples += samples

                if not training:
                    preds_list.extend(preds.cpu().numpy())
                    labels_list.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * total_correct / total_samples

        return (avg_loss, accuracy, (preds_list, labels_list)) if not training else (avg_loss, accuracy, None)

    def save_history(self):
        history_path = os.path.join(self.save_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=4)
        logging.info(f"Training history saved to {history_path}")

    def train(self, epochs, patience, model_path=None):
        model_path = model_path or os.path.join(self.save_dir, "best_model.pth")
        best_acc, patience_counter = 0, 0

        for epoch in range(epochs):
            train_loss, train_acc, _ = self._evaluate(self.train_loader, training=True)
            val_loss, val_acc, _ = self._evaluate(self.val_loader)

            self.scheduler.step(val_loss)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            logging.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
            )

            self.save_history()

            # Model checkpointing
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                try:
                    torch.save(self.model.state_dict(), model_path)
                    logging.info(f"New best model saved to {model_path}")
                except Exception as e:
                    logging.error(f"Failed to save model: {e}")
                    
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info("Early stopping triggered.")
                    break

    def test(self, model_path=None):
        model_path = model_path or os.path.join(self.save_dir, "best_model.pth")
        self.model.load_state_dict(torch.load(model_path))
        logging.info(f"Model loaded from {model_path} for testing.")

        test_loss, test_acc, (preds, labels) = self._evaluate(self.test_loader)
        logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

        class_names = [f"Class {i}" for i in range(self.num_classes)]

        print("\nClassification Report:")
        report = classification_report(labels, preds, target_names=class_names, output_dict=True)
        print(classification_report(labels, preds, target_names=class_names))

        cm = confusion_matrix(labels, preds)
        cm_path = os.path.join(self.save_dir, "confusion_matrix.png")

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        
        plt.savefig(cm_path)
        plt.close()
        logging.info(f"Confusion matrix saved to {cm_path}")

        # Prepare and save test metrics
        test_metrics = {
            'accuracy': test_acc / 100,
            'loss': test_loss,
            'report': report
        }

        metrics_path = os.path.join(self.save_dir, "test_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=4)
        logging.info(f"Test metrics saved to {metrics_path}")

        return test_metrics

if __name__ == "__main__":
    from model import Conformer
    from dataset import BCI2aDataset
    
    model = Conformer()

    train_dataset = BCI2aDataset("../BCICIV_2a/stage1_eog/", mode='train')
    eval_dataset = BCI2aDataset("../BCICIV_2a/stage1_eog/", mode='eval')

    # Create model
    model = Conformer()

    # If you want to split the training data further
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    trainer = Trainer(
        model=model,
        train_dataset=train_subset,
        val_dataset=val_subset,
        test_dataset=eval_dataset,  # Use the eval dataset as test dataset
        batch_size=16,
        num_classes=4,
        save_dir="./results/bci/"
    )
    
    trainer.train(epochs=100, patience=7)
    
    # Test the model
    trainer.test(model_path=os.path.join(trainer.save_dir, 'best_model.pth'))
    print(f"Training complete. Results saved to {trainer.save_dir}")

        
'''   
Stratified Fold:  
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, temp_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n=== Fold {fold + 1} ===")

        # Split remaining data into validation and test sets
        temp_labels = labels[temp_idx]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=fold)
        val_idx_relative, test_idx_relative = next(sss.split(np.zeros(len(temp_labels)), temp_labels))

        val_idx = temp_idx[val_idx_relative]
        test_idx = temp_idx[test_idx_relative]

        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx)

        # New model for each fold
        model = Conformer()

        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=32,
            num_classes=2,
            save_dir=f"./results/CHB/fold{fold + 1}"
        )

        trainer.train(epochs=100, patience=5)
        metrics = trainer.test()
        print(f"Fold {fold + 1} complete. Accuracy: {metrics['accuracy']:.4f}")
'''