# assignment2.py
# ------------------------------------------------------------
# Final Random Forest Model for GitHub Classroom Autograder
# ------------------------------------------------------------

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# ------------------ Load local CSVs ------------------
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
    for c in ["id", "DateTime"]:
        if c in df.columns:
            df.drop(columns=c, inplace=True)
    return df

train = feature_engineer(train)
test = feature_engineer(test)

# ------------------ Split target ------------------
y = train["meal"].astype(int)
X = train.drop(columns=["meal"])

# Handle missing values
imputer = SimpleImputer(strategy="most_frequent")
X_imputed = imputer.fit_transform(X)

# ------------------ Define unfitted model ------------------
model = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# ------------------ Fit model ------------------
modelFit = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
modelFit.fit(X_imputed, y)

# ------------------ Predict ------------------
for col in X.columns:
    if col not in test.columns:
        test[col] = 0
test = test[X.columns]

X_test = imputer.transform(test)
pred = modelFit.predict(X_test)

# Fix dtype issue: convert np.int64 → Python int
pred = [int(x) for x in pred]


