# assignment2.py
# ------------------------------------------------------------
# Random Forest Model - Local Version for Autograder
# ------------------------------------------------------------

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# ------------------ Load Data ------------------
train = pd.read_csv("assignment2train.csv")
test = pd.read_csv("assignment2test.csv")

# ------------------ Feature Engineering ------------------
def feature_engineer(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if "DateTime" in df.columns:
        dt = pd.to_datetime(df["DateTime"], errors="coerce")
        df["hour"] = dt.dt.hour
        df["dow"] = dt.dt.dayofweek
        df["month"] = dt.dt.month
    for col in ["id", "DateTime"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    return df

train = feature_engineer(train)
test = feature_engineer(test)

# ------------------ Split Features and Target ------------------
y = train["meal"].astype(int)
X = train.drop(columns=["meal"])

# Handle missing values
imputer = SimpleImputer(strategy="most_frequent")
X_imputed = imputer.fit_transform(X)

# ------------------ Define Model (Required Name) ------------------
def model(random_state=42):
    return RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )

# ------------------ Fit Model (Required Name) ------------------
modelFit = model()
modelFit.fit(X_imputed, y)

# ------------------ Predict (Required Name) ------------------
for col in X.columns:
    if col not in test.columns:
        test[col] = 0
test = test[X.columns]

X_test = imputer.transform(test)
pred = pd.Series(modelFit.predict(X_test), name="pred")
