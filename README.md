# 🧠 Brain Tumor Classification Using Xception and Transfer Learning

[![Python](https://img.shields.io/badge/python-3.10-blue.svgimg.shields.io/badge/tensorense: MIT](https://img.shields.io/badge/license-MIT-green.svg deep learning pipeline utilizing **Xception** for multiclass classification of brain tumors from MRI images—including data preprocessing, augmentation, transfer learning, fine-tuning, comprehensive evaluation, and visualization. This pipeline is optimized for use on GPUs and is compatible with deployment platforms like Gradio and Hugging Face Spaces.

***

## 🚀 Project Overview

This project aims to assist medical professionals by providing automated, high-accuracy detection and classification of brain tumors in MRI scans. The model uses the Xception architecture pretrained on ImageNet and classifies images into four categories:

- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **No Tumor**

***

## 📂 Dataset

- **Source:** [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Total Images:** 7,023  
  - **Training:** 4,571
  - **Validation:** 1,141 (20% split from training set)
  - **Testing:** 1,311
- **Classes:** `glioma`, `meningioma`, `pituitary`, `notumor`
- **Image Size:** 224×224 pixels
- **Preprocessing:** All images are normalized to the  range

***

## 🖼️ Data Preprocessing and Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

val_test_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

val_data = val_test_datagen.flow_from_directory(
    train_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
    subset='validation'
)

test_data = val_test_datagen.flow_from_directory(
    test_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
```

- **Augmentations:** Random rotations (±30°), width/height shifts (±20%), shear, zoom, horizontal flips
- **Batch Size:** 32

***

## ⚖️ Class Weights Calculation

To address class imbalance:

```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_data.classes), y=train_data.classes)
class_weights_dict = dict(enumerate(class_weights))
print(class_weights_dict)
# Output: {0: 1.0811, 1: 1.0660, 2: 0.8956, 3: 0.9801}
```

***

## 🏗️ Model Architecture

```python
from tensorflow.keras import layers, models
from tensorflow.keras.applications import Xception

base_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='silu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='silu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(256, activation='silu'),
    layers.Dropout(0.3),
    layers.Dense(4, activation='softmax')
])

model.summary()
```

- **Total params:** ~23.6M
- **Trainable params:** ~2.75M (with base frozen)
- **Non-trainable params:** ~20.86M

***

## ⚙️ Training Setup

```python
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, TopKCategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy', AUC(name='auc'), TopKCategoricalAccuracy(k=1, name='top_1_accuracy')]
)
early_stopper = EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)

# Phase 1: Feature extraction
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    batch_size=32,
    callbacks=[early_stopper],
    verbose=1,
    class_weight=class_weights_dict
)

# Phase 2: Fine-tuning
base_model.trainable = True
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy', AUC(name='auc'), TopKCategoricalAccuracy(k=1, name='top_1_accuracy')]
)
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    batch_size=32,
    callbacks=[early_stopper],
    verbose=1,
    class_weight=class_weights_dict
)
```

***

## 📊 Evaluation and Metrics

### Test and Validation Metrics

```python
test_loss, test_accuracy, test_auc, test_top1 = model.evaluate(test_data, verbose=1)
print(f"\nTest Accuracy: {test_accuracy*100:.4f}%")
print(f"Test AUC: {test_auc*100:.4f}%")
print(f"Top-1 Test Accuracy: {test_top1*100:.4f}%")

val_loss, val_accuracy, val_auc, val_top1 = model.evaluate(val_data, verbose=1)
print(f"\nValidation Accuracy: {val_accuracy*100:.4f}%")
print(f"Validation AUC: {val_auc*100:.4f}%")
print(f"Top-1 Validation Accuracy: {val_top1*100:.4f}%")
```

- **Test Accuracy:** 93.59%
- **Test AUC:** 99.24%
- **Validation Accuracy:** 90.62%
- **Validation AUC:** 98.72%

### Classification Report and Confusion Matrix

```python
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(test_data)
y_pred_class = np.argmax(y_pred, axis=1)
y_true = test_data.classes

class_labels = list(val_data.class_indices.keys())
print(classification_report(y_true, y_pred_class, target_names=class_labels))

cm = confusion_matrix(y_true, y_pred_class)
```

#### Classification Report

| Class      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| Glioma     | 0.93      | 0.91   | 0.92     | 300     |
| Meningioma | 0.92      | 0.85   | 0.88     | 306     |
| Notumor    | 0.99      | 0.98   | 0.99     | 405     |
| Pituitary  | 0.89      | 0.99   | 0.94     | 300     |

- **Overall Accuracy:** 0.9359
- **Weighted Precision:** 0.9366
- **Weighted Recall:** 0.9359
- **Weighted F1:** 0.9354

#### Confusion Matrix

Visualized with seaborn:

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
```
***

## 📈 Training Curves

Save and visualize training/validation accuracy and loss curves:

```python
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Train Acc')
plt.plot(epochs_range, val_acc, label='Val Acc')
plt.legend(loc='lower right')
plt.title('Accuracy over Epochs')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.legend(loc='upper right')
plt.title('Loss over Epochs')

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()
```

***

## 💾 Model Saving

```python
from tensorflow.keras.models import save_model
save_model(model, 'brain_tumor_Xception.h5')
```

***

## 💡 Deployment
The model working can be found  at this (https://huggingface.co/spaces/kvamshi04/brain-tumor-xception)
***

## 📝 Environment Setup

- Python 3.10
- TensorFlow 2.18 (or compatible)
- GPU recommended (for speed; tested on Tesla P100 16GB)
- Required libraries: `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

Set environment variables for optimized memory usage (when running on Kaggle or Colab):

```python
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
```

***

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

***

## 🙏 Acknowledgements

- [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- TensorFlow and Keras
- Scikit-learn
- Gradio and Hugging Face Spaces
***
