# Handwritten-Digit-Recognition-using-Decision-Trees-Random-Forest

This project aims to classify handwritten digits using two powerful machine learning algorithms: **Decision Trees** and **Random Forests**. I used the popular **Digits dataset** from `scikit-learn`, which contains 8x8 images of digits from 0 to 9. The goal is to predict the correct digit from these grayscale images.

---

## Overview

Handwritten digit recognition is a classic problem in the field of computer vision and machine learning. This project builds models to recognize digits using:
- **Decision Tree Classifier**
- **Random Forest Classifier**

Each image in the dataset is represented as a flattened array of pixel values, and each model is trained to learn patterns that correspond to individual digits.

---

## Dataset

The project uses the built-in `load_digits()` dataset from `sklearn.datasets`, which includes:
- 1797 total samples
- 10 digit classes (0 to 9)
- 64 features (representing 8x8 grayscale image pixels)

---

## Technologies Used

- Python 
- Scikit-learn (`sklearn`)
- Matplotlib & Seaborn for data visualization
- NumPy & Pandas for data manipulation

---

## Models Used

### Decision Tree
- A simple tree-like structure that splits data based on feature thresholds.
- Great for interpretability, but prone to overfitting on complex datasets.

### Random Forest
- An ensemble of multiple Decision Trees.
- Uses **bagging** to reduce overfitting and increase accuracy.
- More robust and generalizes better than a single tree.

---

## Evaluation Metrics

I used the following metrics to evaluate our models:

- **Accuracy**: Overall correctness of the model.
- **Precision**: Out of all predicted digits, how many were correct.
- **Recall**: Out of all actual digits, how many were correctly predicted.
- **F1-Score**: Harmonic mean of Precision and Recall.
- **Confusion Matrix**: Detailed view of true vs. predicted classifications.

---

## Results

1. Accuracy Comparison:
Decision Tree Accuracy: 84.17%
Random Forest Accuracy: 97.22%

Observation:
Random Forest provides a much higher accuracy than a single Decision Tree, meaning it can identify handwritten digits more correctly in most cases.

2. Precision, Recall, and F1-Score Analysis:
Decision Tree shows inconsistent performance across classes, with some digits like '8' and '9' having noticeably lower precision and recall.

Random Forest gives high and balanced scores across all classes, meaning it's more reliable for every digit.

3. Model Stability:
Decision Tree is a single tree, so it’s more prone to overfitting or underfitting based on how the data is split.

Random Forest uses bagging (bootstrap aggregation) to combine many trees, which:

Reduces variance

Handles noise better

Generalizes much more effectively

Conclusion:
Random Forest is a superior choice for handwritten digit classification in this project. It not only achieves higher accuracy but also provides consistent and reliable predictions across all digit classes. While Decision Trees are simpler and easier to interpret, Random Forest’s ensemble approach makes it much more powerful and suitable for real-world applications.



