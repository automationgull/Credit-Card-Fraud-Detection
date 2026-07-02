# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv(r"D:\Pycharm\Fraude-Detector.Project\data\creditcard.csv")

# Check the data
print(df.head())
print(df.describe())
print(df.info())

# Check missing values
print(df.isnull().sum())

# Check class distribution
print(df["Class"].value_counts())

# Check duplicate values
print(df.duplicated().sum())

# Clean duplicates
df = df.drop_duplicates()
print(df.shape)

# Split data into features and target
X = df.drop("Class", axis=1)
Y = df["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

# Scaling only Time and Amount columns
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[["Time", "Amount"]] = scaler.fit_transform(
    X_train[["Time", "Amount"]]
)

X_test_scaled[["Time", "Amount"]] = scaler.transform(
    X_test[["Time", "Amount"]]
)

print("Preprocessing completed successfully.")
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)