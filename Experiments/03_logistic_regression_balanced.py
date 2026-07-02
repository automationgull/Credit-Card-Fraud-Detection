"""
Experiment 03

Model:
Logistic Regression

Scaling:
StandardScaler

Imbalance Handling:
class_weight="balanced"

Objective:
Improve fraud detection recall by giving higher importance to the minority class.
"""
#import Libraries

import pandas as pd
from sklearn.linear_model import LogisticRegression

#data Upload
df = pd.read_csv(r'D:\Pycharm\Fraude-Detector.Project\data\creditcard.csv')

 # Check the Data
print(df.head())
print(df.describe())
print(df.info())

 #Check missing values
print(df.isnull().sum())

 #Check class distribution
print(df["Class"].value_counts())

 #Check Duplicate values
print(df.duplicated().sum())

 #Clean Duplications
df = df.drop_duplicates()
print(df.shape)

 #Split Data
from sklearn.model_selection import train_test_split
X = df.drop('Class', axis=1)
Y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

 #Model Training and scaling
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[["Time","Amount"]] = scaler.fit_transform(X_train[["Time","Amount"]])
X_test_scaled[["Time","Amount"]] = scaler.transform(X_test[["Time","Amount"]])

model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
model.fit(X_train_scaled, y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
 #Prediction
y_pred = model.predict(X_test_scaled)

print("*" * 50)

print("Baseline Logistic Regression")

print("*" * 50)

 #Accuracy
print("accuracy:",
      accuracy_score(y_test, y_pred))
 #confusion matrix
print("confusion matrix:",
      confusion_matrix(y_test, y_pred))
 #Classification
print("classification_report:",
      classification_report(y_test, y_pred))