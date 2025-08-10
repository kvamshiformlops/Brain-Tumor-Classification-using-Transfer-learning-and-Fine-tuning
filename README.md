
# 🧠 Brain Tumor MRI Classification with Xception

A high-performance deep learning pipeline for **multiclass brain tumor classification** in MRI scans using transfer learning and fine-tuning with the Xception architecture.

***

## 🚀 Project Summary

Classifies brain MRI images into four clinically relevant classes:

- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **No Tumor**

The pipeline leverages TensorFlow, image augmentation, transfer learning, and detailed evaluation—ready for deployment with Gradio or Hugging Face Spaces.

***

## 📂 Dataset and Preprocessing

- **Source:** [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Total Images:** 7,023  
    - Training: 4,571  
    - Validation: 1,141 (20% of train split)  
    - Test: 1,311  
- **Classes:** `glioma`, `meningioma`, `pituitary`, `notumor`
- **Preprocessing:**
    - Images resized to **224×224**
    - Rescaled pixel values to[0,1]
    - Augmentation:  
        - *Rotation* (±30º)  
        - *Width/height shift* (±20%)  
        - *Shear & zoom*  
        - *Horizontal flip*

**Augmentation code sample:**
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
```

**Class Balancing:**  
- `class_weight` computed for training to mitigate imbalance  
    - *Values:* `{0: 1.08, 1: 1.07, 2: 0.90, 3: 0.98}`

***

## 🏗️ Model Architecture

- **Base:** Xception (`include_top=False`, pretrained on ImageNet, frozen during initial training)
- **Custom Head:**
    - `GlobalAveragePooling2D`
    - Dense(1024, activation='silu') → BatchNormalization → Dropout(0.5)
    - Dense(512, activation='silu') → BatchNormalization → Dropout(0.4)
    - Dense(256, activation='silu') → Dropout(0.3)
    - `Dense(4, activation='softmax')`
- **Parameters:**
    - Total: ~23.6M
    - Trainable (frozen base): ~2.75M

## ⚙️ Training Setup

- **Optimizer:** Adam
- **Stages:**
    1. *Feature Extraction*: Xception frozen, `lr=1e-4`, train custom head
    2. *Fine-tuning*: Unfreeze base model, `lr=1e-5`
- **Loss:** `categorical_crossentropy`
- **Metrics:**  
    - `accuracy`  
    - `AUC`  
    - `Top-1 Accuracy`
- **Callbacks:** Early stopping (`patience=5`, restore best weights)
- **Batch Size:** 32  
- **Epochs:** Up to 30 (with early stopping) at feature extraction stage and upto 15 (with early stopping) at fine-tuning stage

***

## 📊 Performance Metrics

**Overall Test Performance:**
- **Accuracy:** 93.6%
- **AUC:** 99.2%
- **Precision (weighted):** 93.7%
- **Recall (weighted):** 93.6%
- **F1-Score (weighted):** 93.5%

**Per-Class Test Breakdown:**
| Class      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| Glioma     | 0.93      | 0.91   | 0.92     | 300     |
| Meningioma | 0.92      | 0.85   | 0.88     | 306     |
| Notumor    | 0.99      | 0.98   | 0.99     | 405     |
| Pituitary  | 0.89      | 0.99   | 0.94     | 300     |

- **Confusion matrix and learning curves** included for interpretability.
- **Validation Accuracy:** 90.6%, **Validation AUC:** 98.7%

***

## 🔗 Live Demo

**Try out the deployed model here:**  
👉 [Hugging Face Spaces App](https://huggingface.co/spaces/kvamshi04/brain-tumor-xception)

Upload an MRI image and receive instant classification and probabilities for each tumor category.

***

## 📄 License

MIT License (see `LICENSE`).

***

## 🙏 Credits

- Kaggle dataset providers
- TensorFlow/Keras,Scikit-learn
- Gradio,Hugging Face Spaces

***

[1] https://huggingface.co/spaces/kvamshi04/brain-tumor-xception
