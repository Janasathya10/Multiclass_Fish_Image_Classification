from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import load_data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

train_gen, val_gen, test_gen = load_data()
class_indices = test_gen.class_indices
target_names = list(class_indices.keys())
label_ids = list(class_indices.values())

model_files = [f for f in os.listdir('models') if f.endswith('.h5')]
results = []

for file in model_files:
    path = f'models/{file}'
    model = load_model(path)
    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    report = classification_report(
        y_true, y_pred,
        labels=label_ids,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    acc = report['accuracy']
    f1 = np.mean([report[class_name]['f1-score'] for class_name in target_names])
    results.append({'Model': file, 'Accuracy': acc, 'F1-Score': f1})

    # Optional: Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=False, xticklabels=target_names, yticklabels=target_names, fmt='d')
    plt.title(file)
    plt.savefig(f'models/{file.replace(".h5", "_cm.png")}')
    plt.clf()

df = pd.DataFrame(results)
df.to_csv('models/leaderboard.csv', index=False)
print(df)