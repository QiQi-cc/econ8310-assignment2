# assignment2.py
# Random Forest 

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import numpy as np

# 1) Load local CSVs 
train = pd.read_csv("assignment2train.csv")
test  = pd.read_csv("assignment2test.csv")

# 2) Feature engineering
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Parse datetime -> calendar features
    if "DateTime" in df.columns:
        dt = pd.to_datetime(df["DateTime"], errors="coerce")
        df["hour"]  = dt.dt.hour
        df["dow"]   = dt.dt.dayofweek  
        df["month"] = dt.dt.month
        # Extra helpful bins
        # breakfast: 6–10, lunch: 11–14, afternoon: 15–17, dinner: 18–21
        df["is_breakfast"] = df["hour"].between(6, 10).astype(int)
        df["is_lunch"]     = df["hour"].between(11, 14).astype(int)
        df["is_afternoon"] = df["hour"].between(15, 17).astype(int)
        df["is_dinner"]    = df["hour"].between(18, 21).astype(int)
        df["is_weekend"]   = df["dow"].isin([5, 6]).astype(int)  

    # Monetary stabilizers (robust to missing)
    for col in ["Total", "Discounts"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Total" in df.columns:
        # caps to reduce outliers effect
        df["Total_capped"] = df["Total"].clip(lower=0, upper=df["Total"].quantile(0.99))
        df["is_total_gt5"] = (df["Total"] >= 5).astype(int)
        df["is_total_gt8"] = (df["Total"] >= 8).astype(int)

    if "Discounts" in df.columns:
        df["Discounts_capped"] = df["Discounts"].clip(lower=0, upper=df["Discounts"].quantile(0.99))

    # Drop id / raw DateTime
    for c in ("id", "DateTime"):
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    return df

train = feature_engineer(train)
test  = feature_engineer(test)

# 3) Split
y = train["meal"].astype(int)
X = train.drop(columns=["meal"])

# 4) Impute
imputer = SimpleImputer(strategy="most_frequent")
X_imputed = imputer.fit_transform(X)

# 5) Define model (UNFITTED). Keep as a top-level variable.
model = RandomForestClassifier(
    n_estimators=900,
    max_depth=None,
    max_features="sqrt",
    min_samples_leaf=1,
    bootstrap=True,
    class_weight=None,       
    random_state=42,
    n_jobs=-1
)

# 6) Fit THE SAME INSTANCE and expose it as modelFit
modelFit = model.fit(X_imputed, y)

# 7) Align test columns to train
for col in X.columns:
    if col not in test.columns:
        test[col] = 0

test = test[X.columns]

X_test = imputer.transform(test)

# 8) Predict 
pred = [int(x) for x in modelFit.predict(X_test)]







