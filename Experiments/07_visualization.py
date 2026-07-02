""" File # 7
    Title: Visualization of Random Forest Model
    Author: Faiqua Gull
    Date: 2020-05-29
    Objective: To show the different values of trained model at different stages"""
#import libraries

import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import f1_score
from sklearn.metrics._plot import confusion_matrix

#load data
df = pd.read_csv(r'D:\Pycharm\Fraude-Detector.Project\data\creditcard.csv')
print(df.head())
print(df.shape)

# Graph 1: Class Distribution

class_counts = df["Class"].value_counts()

plt.figure(figsize=(6, 4))
plt.bar(["Genuine", "Fraud"], class_counts.values)

plt.title("Class Distribution")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")

plt.savefig(r"D:\Pycharm\Fraude-Detector.Project\screenshots/class_distribution.png")
plt.show()

#Graph 2: Model comparison

models = ["Logistic", "Balanced","SMOTE","RandomForest"]
f1_score = [0.67, 0.11, 0.10, 0.82]
plt.figure(figsize=(8, 6))
plt.bar(models, f1_score)
plt.title("Model Comparison by F1-score", loc="center",color="red")

plt.xlabel("Models", color = "blue")
plt.ylabel("F1 Score", color = "blue")

plt.savefig(r"D:\Pycharm\Fraude-Detector.Project\screenshots\F1-score.png")
plt.show()

#Graph 3 : Confusion Matrix for Random Forest

confusion_values = [[56647, 4],
                    [26, 69]]
plt.figure(figsize=(6, 5))
plt.imshow(confusion_values, cmap="Blues")
plt.title("Confusion Matrix - Random Forest", loc="center",color="red")
plt.xlabel("Predicted Class", color = "blue")
plt.ylabel("Actual Class", color = "blue")
plt.xticks([0,1],["Genuine", "Fraud"])
plt.yticks([0,1],["Genuine","Fraud"])
for i in range(2):
    for j in range(2):
        plt.text(j,i, confusion_values[i][j], ha="center", va="center", color="red")

plt.savefig(r"D:\Pycharm\Fraude-Detector.Project\screenshots\confusion_matrix.png")
plt.show()

# Graph 4: Feature Importance - Random Forest

#load saved random forest model
model = joblib.load(r"D:\Pycharm\Fraude-Detector.Project\models\random_forest_model.pkl")

# Prepare feature names
X = df.drop("Class", axis=1)
feature_names = X.columns

# Get feature importance values from model
importance_values = model.feature_importances_

# Create DataFrame for feature importance
feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

# Sort features by importance
feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Select top 10 important features
top_10_features = feature_importance_df.head(10)

# Plot feature importance
plt.figure(figsize=(8, 5))
plt.bar(top_10_features["Feature"], top_10_features["Importance"])

plt.title("Top 10 Feature Importances - Random Forest")
plt.xlabel("Features")
plt.ylabel("Importance Score")

plt.savefig(r"D:\Pycharm\Fraude-Detector.Project\screenshots/feature_importance_random_forest.png")
plt.show()