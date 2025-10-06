# Random Forest

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Step 1: Load data 
train = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/econ8310-assignment2/main/assignment2train.csv")
test  = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/econ8310-assignment2/main/assignment2test.csv")

# Step 2: Feature engineering
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if "DateTime" in df.columns:
        dt = pd.to_datetime(df["DateTime"], errors="coerce")
        df["hour"]  = dt.dt.hour
        df["dow"]   = dt.dt.dayofweek
        df["month"] = dt.dt.month

    for c in ("id", "DateTime"):
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    return df

train = feature_engineer(train)
test  = feature_engineer(test)

# Step 3: Split features
y = train["meal"].astype(int)
X = train.drop(columns=["meal"])

# Step 4: Impute 
imputer = SimpleImputer(strategy="most_frequent")
X_imputed_array = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed_array, columns=X.columns, index=X.index)  

# Step 5: Define model and fit THE SAME INSTANCE
model = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
modelFit = model.fit(X_imputed, y)  

# Step 6: Align test columns and impute 
for col in X.columns:
    if col not in test.columns:
        test[col] = 0
test = test[X.columns]

X_test_array = imputer.transform(test)
X_test = pd.DataFrame(X_test_array, columns=test.columns, index=test.index)  

# Step 7: Predict 
pred = [int(x) for x in modelFit.predict(X_test)]

# Quick preview (optional)
pd.DataFrame({"pred": pred}).head()













