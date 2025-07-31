
# Dogs vs Cats Image Classification (Custom CNN & Fine-Tuned VGG16)

This project presents a comparative analysis between two deep learning approaches for binary image classification—distinguishing between dogs and cats. The models implemented include a **Custom Convolutional Neural Network (CNN)** and a **Fine-Tuned VGG16** model using transfer learning.

---

## Dataset Overview
- **Source**: [Kaggle - Dogs vs. Cats](https://www.kaggle.com/datasets/biaiscience/dogs-vs-cats)
- **Structure**: A curated subset of 5,000 images (from 25,000 original)
  - 2,000 for training (1,000 cats + 1,000 dogs)
  - 1,000 for validation (500 cats + 500 dogs)
  - 2,000 for testing (1,000 cats + 1,000 dogs)

---

## Objective
To develop and evaluate two deep learning models that classify input images as either **Cat** or **Dog**, and to compare their performance using established metrics.

---

## Workflow Summary

### 1. Data Preparation
- Organized directory structure for train/validation/test sets.
- Visualized class samples to verify dataset integrity.

### 2. Data Augmentation
- Applied techniques: horizontal flip, zoom, rotation, width/height shift.
- Implemented via Keras `ImageDataGenerator`.

### 3. Model Development
- **Model 1**: Custom CNN with convolutional, pooling, and dropout layers.
- **Model 2**: Pre-trained VGG16 with top layers removed, custom dense layers added.

### 4. Training Configuration
- Training using augmented data.
- EarlyStopping and ModelCheckpoint callbacks to retain the best performing model.

### 5. Evaluation
- Accuracy, confusion matrix, precision, recall, F1-score.
- Precision-Recall (PR) curves.
- Misclassification analysis to investigate prediction errors.

---

## Performance Comparison

| Metric           | Custom CNN | Fine-Tuned VGG16 |
|------------------|------------|------------------|
| Accuracy         | ~83%       | ~96%             |
| PR AUC           | ~0.84      | ~0.96            |
| Overfitting      | Moderate   | Well-controlled  |
| Training Time    | Faster     | Longer           |
| Misclassifications | More     | Fewer            |

> **Conclusion**: The Fine-Tuned VGG16 model demonstrated superior accuracy, robustness, and generalization. Transfer learning proved highly effective, especially when working with limited training data.

---

## Tools and Libraries
- Python 3.x
- TensorFlow / Keras
- Scikit-learn
- Matplotlib / NumPy

---

## Repository Structure

```
├── Data/
│   ├── train/               # Full original dataset (25,000 images)
│   └── cats_vs_dogs/        # Filtered and structured dataset (train/val/test)
├── models/                  # Best model weights saved via callbacks
├── notebooks/               # Jupyter Notebooks with step-by-step analysis
├── README.md/               # Project documentation
└── requirements.txt 
```

---

## Author
**Mandeep Singh Brar**  
 Student, CSCN8010

---

## Note
The dataset folder is excluded from version control (`.gitignore`) due to its size. Please download and extract the dataset manually into `Data/train/` before executing the notebooks.

---
