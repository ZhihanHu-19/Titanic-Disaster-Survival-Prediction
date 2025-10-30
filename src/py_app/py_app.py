# packages import
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# Step 1: Load data
print("\nStep 1: Load data")
train_path = os.environ.get("TRAIN_PATH", "src/data/train.csv")
test_path  = os.environ.get("TEST_PATH", "src/data/test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
# see the overview of the DataFrame
print(f"Train shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")
print("Train head:")
print(train_df.head(3))

# Step 2: Process data
print("\nStep2: Process data")
# Drop rows with missing target since it will not be useful to train the model
train_df = train_df.dropna(subset=["Survived"])

# Fill missing values
train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
test_df["Age"] = test_df["Age"].fillna(test_df["Age"].median())
train_df["Embarked"] = train_df["Embarked"].fillna(train_df["Embarked"].mode()[0])
test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].median())

# Encode categorical columns
for col in ["Sex", "Embarked"]:
    le = LabelEncoder()
    le.fit(pd.concat([train_df[col], test_df[col]], axis=0))
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

# Select features
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = train_df[features]
y = train_df["Survived"]

# Standardize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(test_df[features])

print(f"Features used: {features}")

# Step 3: Train the model
print("\nStep 3: Train the model")
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled,y)
train_preds = model.predict(X_scaled)
train_accuracy = accuracy_score(y, train_preds)
print(f"Training accuracy: {train_accuracy:.4f}")

# Step 4: Predict test
print("\nStep 4: Predict Test")
test_preds = model.predict(X_test_scaled)
print(f"First 10 test predictions: {test_preds[:10]}")

# Step 5: Save submission
print("\nStep 5: Save predictions")
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": test_preds
})
os.makedirs("src/data", exist_ok=True)
submission.to_csv("src/data/predictions_py.csv", index=False)
print("Saved predictions to src/data/predictions_py.csv")