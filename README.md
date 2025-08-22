# Multiclass_Fish_Image_Classification Project

This project aims to classify fish species using Convolutional Neural Networks (CNNs) and Transfer Learning techniques. It includes training and evaluation scripts, as well as a Streamlit web application for prediction.

---

## 📁 Project Structure
```
fish-classification/
├── app.py             # Streamlit web app for model inference
├── cnn_model.py       # Custom CNN architecture
├── data_loader.py     # Data generators for train/val/test sets
├── evaluate.py        # Evaluate all models and save metrics
├── train_model.py     # Orchestrates training (CNN + Transfer Models)
├── transfer_model.py  # Pretrained backbones & transfer model builder
├── models/            # Trained .h5 models + metadata JSONs
├── results/           # Evaluation results (reports, confusion matrices)
├── data/              # Dataset (train/val/test folders with classes)
│ ├── train/
│ ├── val/
│ └── test/
└── requirements.txt   # Dependencies
```

---

## 📦 Features

✅ Custom CNN training  
✅ Transfer learning with **VGG16, ResNet50, InceptionV3, MobileNet, EfficientNetB0**  
✅ Metadata saving for consistent preprocessing and label mapping  
✅ Model evaluation with **accuracy, F1 score, confusion matrix, classification report**  
✅ Streamlit UI for real-time fish image classification  
✅ Leaderboard CSV for model comparison  

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/fish-classification.git
cd fish-classification
```

### 2. Create and Activate a Virtual Environment (optional but recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📁 Data Format

Place your dataset inside the `data/` directory, structured like:

```
data/
├── train/
│   ├── fish_class1/
│   ├── fish_class2/
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

- Each subfolder = one fish species.
- Images can be .jpg, .png, etc.
- Train set → model learning, Val set → hyperparameter tuning, Test set → final evaluation.

---

## 🚀 Training Models

### 1. Train Custom CNN

```bash
python train_model.py

or you can train each model alone by using (python train_model.py --only CNN)
```

## 📊 Evaluate Models

To compare all trained models and generate confusion matrices:

```bash
python evaluate.py
```

- results/leaderboard.csv → model comparison (accuracy, precision, recall, F1)
- Per-model classification_report.csv
- Per-model confusion_matrix.csv

---

## 🌐 Run Streamlit Web App

```bash
streamlit run app.py
```

Then open the link (usually http://localhost:8501) in your browser.

---

## 📉 Output Samples

- results/leaderboard.csv → Accuracy & F1-score comparison of all models.
- results/*classification_report.csv → per-model metrics.
- results/*confusion_matrix.csv → confusion matrix for each model.

models/*.h5 + *_metadata.json → saved models with metadata.
---

## 🧠 Models Used

✅ Custom CNN (from cnn_model.py)

✅ Transfer Learning (from transfer_model.py):

- VGG16
- ResNet50
- InceptionV3
- MobileNet
- EfficientNetB0

---

## 📝 Requirements

Typical `requirements.txt` includes:

```
tensorflow==2.17.1
numpy==1.24.4
pandas==1.5.3
matplotlib==3.7.1
scikit-learn==1.2.2
opencv-python==4.8.0.76
streamlit==1.33.0
seaborn==0.12.2
pillow==9.5.0
```
------------------------------------------------------------------------------------------------
