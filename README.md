# 🐟 Multiclass_Fish_Image_Classification Project

This project aims to classify fish species using Convolutional Neural Networks (CNNs) and Transfer Learning techniques. It includes training and evaluation scripts, as well as a Streamlit web application for prediction.

---

## 📁 Project Structure

```
fish-classification/
├── app.py                # Streamlit web app for model inference
├── cnn_model.py          # Custom CNN architecture
├── data_loader.py        # Data generators for train/val/test sets
├── evaluate.py           # Evaluate all models and save metrics
├── train_model.py        # Train the custom CNN model
├── transfer_model.py     # Train multiple transfer learning models
├── models/               # Trained .h5 models and plots
├── data/                 # Folder structure with train/val/test images
│   ├── train/
│   ├── val/
│   └── test/
└── requirements.txt      # Dependencies
```

---

## 📦 Features

- ✅ Custom CNN training
- ✅ Transfer learning with **VGG16, ResNet50, InceptionV3, MobileNet, EfficientNetB0**
- ✅ Model evaluation with accuracy, F1 score, confusion matrix
- ✅ Streamlit UI for fish image classification
- ✅ Leaderboard CSV for model comparison

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

Each class should contain images (JPG/PNG) of that fish type.

---

## 🚀 Training Models

### 1. Train Custom CNN

```bash
python train_model.py
```

### 2. Train Transfer Learning Models

```bash
python transfer_model.py
```

This will save models under the `models/` directory.

---

## 📊 Evaluate Models

To compare all trained models and generate confusion matrices:

```bash
python evaluate.py
```

This creates a `leaderboard.csv` and confusion matrix plots inside the `models/` folder.

---

## 🌐 Run Streamlit Web App

```bash
streamlit run app.py
```

Then open the link (usually http://localhost:8501) in your browser.

---

## 📉 Output Samples

- `models/leaderboard.csv`: Accuracy and F1 score comparison
- `models/*.png`: Confusion matrices & accuracy plots

---

## 🧠 Models Used

- ✅ Custom CNN (`cnn_model.py`)
- ✅ Pretrained Models:
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
```
------------------------------------------------------------------------------------------------
