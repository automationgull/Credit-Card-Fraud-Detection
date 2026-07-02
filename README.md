# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using multiple classification techniques and model evaluation metrics.

## Project Overview

This project focuses on detecting credit card fraud using a highly imbalanced dataset. The main goal was not only to train a model, but to compare different approaches and understand which model performs better for fraud detection.

## Dataset

The dataset contains anonymized transaction features:

* `Time`
* `V1` to `V28`
* `Amount`
* `Class`

Target column:

* `0` = Genuine Transaction
* `1` = Fraudulent Transaction

The dataset is highly imbalanced, with genuine transactions much higher than fraudulent transactions.

## Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn
* Imbalanced-learn
* Joblib
* Git & GitHub

## Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── Experiments/
│   ├── 01_data_preprocessing.py
│   ├── 02_logistic_regression_baseline.py
│   ├── 03_logistic_regression_balanced.py
│   ├── 04_logistic_regression_smote.py
│   ├── 05_random_forest.py
│   ├── 06_load_model.py
│   └── 07_visualization.py
│
├── models/
│   ├── random_forest_model.pkl
│   └── scaler.pkl
│
├── screenshots/
│   ├── class_distribution.png
│   ├── F1-score.png
│   ├── confusion_matrix.png
│   └── feature_importance_random_forest.png
│
├── README.md
└── .gitignore
```

## Experiments Performed

### 1. Logistic Regression Baseline

A basic Logistic Regression model was trained after scaling `Time` and `Amount`.

### 2. Logistic Regression with Class Weight

Because the dataset was highly imbalanced, `class_weight="balanced"` was used to improve fraud detection recall.

### 3. Logistic Regression with SMOTE

SMOTE was applied to the training data to create synthetic fraud examples and balance the minority class.

### 4. Random Forest Classifier

A Random Forest model was trained and selected as the best overall model based on precision, recall, and F1-score.

## Model Comparison

| Model               | Accuracy | Precision | Recall | F1-score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |   99.91% |      0.88 |   0.54 |     0.67 |
| Logistic + Balanced |   97.76% |      0.06 |   0.89 |     0.11 |
| Logistic + SMOTE    |   97.37% |      0.05 |   0.87 |     0.10 |
| Random Forest       |   99.95% |      0.95 |   0.73 |     0.82 |

## Best Model

The best model was:

```text
Random Forest Classifier
```

It gave the best overall balance between precision and recall.

## Final Random Forest Result

```text
Accuracy: 99.95%

Confusion Matrix:
[[56647     4]
 [   26    69]]
```

## Visualizations

The project includes:

* Class distribution graph
* Model comparison graph
* Confusion matrix
* Feature importance graph

## Model Saving

The trained Random Forest model and scaler were saved using `joblib`.

```python
joblib.dump(model, "models/random_forest_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
```

The saved model and scaler were also loaded successfully for future prediction use.

## Key Learning

Through this project, I learned:

* Data preprocessing
* Handling imbalanced datasets
* Train-test split with stratification
* Feature scaling
* Logistic Regression
* Class weight balancing
* SMOTE oversampling
* Random Forest Classifier
* Model evaluation using precision, recall, F1-score, and confusion matrix
* Model saving and loading with Joblib
* Data visualization with Matplotlib
* GitHub project publishing

## Future Improvements

* Build a Streamlit web app for CSV upload and prediction
* Try XGBoost or LightGBM
* Add ROC-AUC curve
* Improve model threshold tuning
* Deploy the project online
