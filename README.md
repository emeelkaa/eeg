# Multi-Scale EEG Feature Extraction with Attention Mechanism for Seizure Detection

This repository contains the official code for the paper:

**"Multi-Scale EEG Feature Extraction with Attention Mechanism for Seizure Detection"**

## 🧠 Overview

This project implements a deep learning pipeline for seizure detection from EEG signals. The core of the model utilizes multi-scale convolutional blocks and attention mechanisms to extract informative temporal and spatial features.

---

## 📁 Project Structure

```
├── preprocessing/
│   ├── stage0_to_stage1.py             # Preprocess Raw EEG
│   ├── stage1_to_stage2_seizures.py    # Sample seizure segments, apply standard preprocessing
│   ├── stage1_to_stage2_normals.py     # Sample normal (non-seizure) segments, apply standard preprocessing
│   ├── utils.py                        # Utility functions
│
├── model.py                            # EEG model with multi-scale conv + attention
├── dataset.py                          # Dataset class for loading EEG chunks
├── trainer.py                          # Training script using the dataset/model
├── requirements.txt                    # Dependencies
├── README.md                           # You are here!
```

---

## 🔁 Workflow

1. **Preprocess Raw EEG Data**
   ```bash
   python preprocessing/stage0_to_stage1.py
   ```

2. **Sample Seizure Segments**
   ```bash
   python preprocessing/stage1_to_stage2_seizures.py
   ```

3. **Sample Normal Segments**
   ```bash
   python preprocessing/stage1_to_stage2_normals.py
   ```

4. **Train the Model**
   ```bash
   python trainer.py
   ```
   - This uses `model.py` and `dataset.py` to load data and train the network.

---

## ⚙️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## 📌 Notes

- EEG data must be preprocessed in the specified format.
- Configurations such as sampling length, overlap, and chunk count can be set in the config.py file.

---

## 📫 Contact

For questions or collaboration inquiries, please contact [emilkim01@pusan.ac.rk].
