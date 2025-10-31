# packages import
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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

def mode_from_series(s):
    s_no_na = s.dropna()
    if s_no_na.empty:
        return None
    return s_no_na.mode().iloc[0]

train_df["Embarked"] = train_df["Embarked"].replace("", np.nan)
test_df["Embarked"] = test_df["Embarked"].replace("",np.nan)

# Find some values used to fill
med_age  = train_df["Age"].median(skipna=True)
med_fare = train_df["Fare"].median(skipna=True)
mode_emb = mode_from_series(train_df["Embarked"])

# def a function to clean data
def prep_data(df):
    df = df.copy()
    if "Age" in df:
        df["Age"] = df["Age"].fillna(med_age)
    if "Fare" in df:
        df["Fare"] = df["Fare"].fillna(med_fare)
    if "Embarked" in df:
        df["Embarked"] = df["Embarked"].fillna(mode_emb)
    return df

train_df = prep_data(train_df)
test_df = prep_data(test_df)

# Select features
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
print(f"Features used: {features}")

# Step 3: Train the model
print("\nStep 3: Train the model")
num_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
cat_cols = ["Sex", "Embarked"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ],
    remainder="drop",
)

clf = LogisticRegression(max_iter=1000, solver="lbfgs")

pipe = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", clf)
])

X_train = train_df[features]
y_train = train_df["Survived"].astype(int)

pipe.fit(X_train,y_train)

train_preds = pipe.predict(X_train)
train_accuracy = accuracy_score(y_train, train_preds)
print(f"Training accuracy: {train_accuracy:.4f}")

# Step 4: Predict test
print("\nStep 4: Predict Test")
X_test = test_df[features]
test_preds = pipe.predict(X_test).astype(int)
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