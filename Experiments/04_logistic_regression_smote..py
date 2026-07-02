"""
from imblearn.over_sampling import SMOTE
from scipy.stats import Logistic

Experiment NO : 4

Model: Logistic Regression

Scaling: Standard scaling on Time and Amount

Imbalance Handling: SMOTE

Objective: Create Fraud examples in training data to improve fraud detection

"""
#import Libraries

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Load data

df = pd.read_csv(r'D:\Pycharm\Fraude-Detector.Project\data\creditcard.csv')
# Examining data
print(df.head())

# check and remove duplication
print(df.duplicated().sum())
df = df.drop_duplicates()

#split data
X = df.drop('Class', axis=1)
Y = df['Class']

#Test_Train_Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size = 0.2,
    random_state = 42,
    stratify = Y)

#Scaler values for time and amount
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[["Time","Amount"]] = scaler.fit_transform(X_train[["Time","Amount"]])

X_test_scaled[["Time","Amount"]] = scaler.transform(X_test[["Time","Amount"]])

#Apply SMOTE on scaled training data
smote = SMOTE(random_state =42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, Y_train)
print("Before SMOTE:")
print(Y_train.value_counts())

print("After SMOTE:")
print(y_train_smote.value_counts())

# Train Logistic Regression model
model = LogisticRegression(
    random_state=42,
    max_iter=1000)
model.fit(X_train_smote, y_train_smote)

#Prediction
y_pred = model.predict(X_test_scaled)


#Evaluation
print("*" * 50)
print("Logistic Regression With SMOTE:")
print("*" * 50)

print("Accuracy:", accuracy_score(Y_test, y_pred))

print("Confusion:", confusion_matrix(Y_test, y_pred))

print("Classification Report:", classification_report(Y_test, y_pred))

