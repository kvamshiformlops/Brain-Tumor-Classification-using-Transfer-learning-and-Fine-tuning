# 🧠 Brain Tumor Classification using Transfer Learning and Fine-Tuning

[![Python](https://img.shields.io/badge/pythonFlow](https://img.shields.io/badge/tensorflow-2.18-orange.svg](https://img.shields.io/badge/license-MIT-green.svg learning pipeline utilizing **Xception** for multiclass classification of brain tumors from MRI images.
Includes **data preprocessing**, **augmentation**, **transfer learning**, **fine-tuning**, **evaluation**, and deployment with **Gradio** on **Hugging Face Spaces**.

## 🚀 Project Overview

A transfer-learning approach using the **Xception** architecture pretrained on ImageNet to classify MRI scans into four categories:

- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **No Tumor**

The goal is to assist medical professionals with automated, high-accuracy detection and classification of brain tumors.

***

## 📂 Dataset

- **Source:** [Kaggle: Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Total images:** 7,023  
  - **Training:** 4,571 images  
  - **Validation:** 1,141 images (20% split)  
  - **Testing:** 1,311 images  
- **Classes (4):** `glioma`, `meningioma`, `pituitary`, `notumor`

Each MRI is resized to **224×224 pixels** and normalized to the  range.[1]

***

## 🖼️ Data Preprocessing & Augmentation

```python
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
    rescale=1./255,
    validation_split=0.2
)
```

- **Rescaling:** pixel values scaled to[1]
- **Augmentations:** rotations (±30°), shifts (±20%), shear, zoom, horizontal flips
- **Batch size:** 32
- **Target size:** 224×224

***

## 🏗️ Model Architecture

A Sequential model built upon the Xception base:

- `Xception` (ImageNet weights, `include_top=False`, input 224×224×3)
- `GlobalAveragePooling2D()`
- `Dense(1024, activation='silu')` → `BatchNorm` → `Dropout(0.5)`
- `Dense(512, activation='silu')` → `BatchNorm` → `Dropout(0.4)`
- `Dense(256, activation='silu')` → `Dropout(0.3)`
- Output: `Dense(4, activation='softmax')`

```python
base_model = Xception(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='silu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation='silu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation='silu'),
    Dropout(0.3),
    Dense(4, activation='softmax')
])
```

- **Total params:** ~23.6M
- **Trainable params (frozen base):** ~2.75M
- **Non-trainable params:** ~20.86M

***

## ⚙️ Training Setup

- **Optimizer:** Adam
- **Phase 1 (feature extraction):** lr=1e-4
- **Phase 2 (fine-tuning):** lr=1e-5
- **Loss:** Categorical Cross-Entropy
- **Metrics:** Accuracy, AUC, Top-1 Accuracy
- **Callbacks:** EarlyStopping (patience=5, restore best weights)
- **Class weights:**
    ```python
    {0: 1.0811, 1: 1.0660, 2: 0.8956, 3: 0.9801}
    ```

***

## 📊 Results & Performance Metrics

|         | Validation | Test      |
|---------|------------|-----------|
| Accuracy| 90.62%     | 93.59%    |
| AUC     | 98.72%     | 99.23%    |
| Precision (W) |        | 93.66%    |
| Recall (W)    |        | 93.59%    |
| F1-Score (W)  |        | 93.54%    |

**Per-class Test Accuracy:**

- **Glioma:** 91%
- **Meningioma:** 85%
- **No Tumor:** 98%
- **Pituitary:** 99%

***

## 💻 Deployment

This model is deployed using Gradio on Hugging Face Spaces.  
You can upload an MRI scan and receive predictions with class probabilities in real time.

🔗 **Live Demo:** [Try on Hugging Face](https://huggingface.co/spaces/kvamshi04/brain-tumor-xception)

***

## 📄 License

Licensed under the MIT License. See `LICENSE` for details.

***

## 🙏 Acknowledgements

- Kaggle Dataset
- TensorFlow
- Gradio
- Hugging Face Spaces

[1] https://img.shields.io/badge/python-3.10-blue.svg
[2] https://www.python.org
[3] https://img.shields.io/badge/tensorflo
