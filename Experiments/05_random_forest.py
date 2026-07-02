"""Experiment NO : 5

Model: Random Forest classification

Scaling: Standard scaling on Time and Amount

Imbalance Handling: first Class_weight= Balanced then SMOTE

Objective: Applying different Model to get good precision and recall"""

# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

#load data

df = pd.read_csv(r"D:\Pycharm\Fraude-Detector.Project\data\creditcard.csv")
print(df.head())

#analyze data
print(df.describe())
print(df.info())

#check duplication
print(df.duplicated().sum())
print(df.isnull().sum())

#clean duplications
df =df.drop_duplicates()

#Split data
X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#scaler values for Time, Amount
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[["Time","Amount"]] = scaler.fit_transform(X_train[["Time","Amount"]])
X_test_scaled[["Time","Amount"]] = scaler.transform(X_test[["Time","Amount"]])

#Train Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)
model.fit(X_train_scaled, y_train)

#prediction
y_pred = model.predict(X_test_scaled)

#Evaluation
print("*" * 50)
print("Random Forest Classifier Evaluation")
print("*" * 50)

print("Classification Report:", classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and scaler
import joblib

joblib.dump(model,'D:/Pycharm/Fraude-Detector.Project/models/random_forest_model.pkl')
joblib.dump(scaler,'D:/Pycharm/Fraude-Detector.Project/models/scaler.pkl')

print("✅ Model and Scaler saved successfully.")