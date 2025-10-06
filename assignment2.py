# assignment2.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Step 1: Load local CSVs
train = pd.read_csv("assignment2train.csv")
test = pd.read_csv("assignment2test.csv")

# Step 2: Feature engineering
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

# Step 3: Prepare X, y
y = train["meal"].astype(int)
X = train.drop(columns=["meal"])

# Step 4: Handle missing values
imputer = SimpleImputer(strategy="most_frequent")
X_imputed = imputer.fit_transform(X)

# Step 5: Define models
model = RandomForestClassifier(
    n_estimators=500, 
    max_features="sqrt", 
    min_samples_leaf=2, 
    random_state=42,
    n_jobs=-1
)

modelFit = RandomForestClassifier(
    n_estimators=500, 
    max_features="sqrt", 
    min_samples_leaf=2, 
    random_state=42,
    n_jobs=-1
).fit(X_imputed, y)

# Step 6: Align test features
for col in X.columns:
    if col not in test.columns:
        test[col] = 0
test = test[X.columns]

# Step 7: Predict and convert to list
X_test = imputer.transform(test)
pred = [int(x) for x in modelFit.predict(X_test)]




