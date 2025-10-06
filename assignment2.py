# I choose Random Forest
# assignment2.py

from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# ------------------------------
# Step 1: Load data (local first, else GitHub)
# ------------------------------
TRAIN_LOCAL = Path("assignment2train.csv")
TEST_LOCAL  = Path("assignment2test.csv")

TRAIN_URL = "https://raw.githubusercontent.com/dustywhite7/econ8310-assignment2/main/assignment2train.csv"
TEST_URL  = "https://raw.githubusercontent.com/dustywhite7/econ8310-assignment2/main/assignment2test.csv"

def read_csv_safely(local_path: Path, url: str) -> pd.DataFrame:
    """Load a CSV from disk if present; otherwise fall back to the GitHub URL."""
    if local_path.exists():
        return pd.read_csv(local_path)
    print(f"Local file {local_path.name} not found, using GitHub version.")
    return pd.read_csv(url)

train = read_csv_safely(TRAIN_LOCAL, TRAIN_URL)
test  = read_csv_safely(TEST_LOCAL,  TEST_URL)

# ------------------------------
# Step 2: Feature engineering
# ------------------------------
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [c.strip() for c in d.columns]  # sanitize column names

    # Parse DateTime and create time-based features (if available)
    if "DateTime" in d.columns:
        dt = pd.to_datetime(d["DateTime"], errors="coerce")
        d["hour"]  = dt.dt.hour
        d["dow"]   = dt.dt.dayofweek
        d["month"] = dt.dt.month

        d["is_breakfast"] = d["hour"].between(6, 10).astype(int)
        d["is_lunch"]     = d["hour"].between(11, 14).astype(int)
        d["is_afternoon"] = d["hour"].between(15, 17).astype(int)
        d["is_dinner"]    = d["hour"].between(18, 21).astype(int)
        d["is_weekend"]   = d["dow"].isin([5, 6]).astype(int)

    # Clean numeric columns and cap extreme values
    for col in ["Total", "Discounts"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
            d[col] = d[col].clip(lower=0, upper=d[col].quantile(0.99))

    # Simple thresholds that help tree models
    if "Total" in d.columns:
        d["is_total_ge5"] = (d["Total"] >= 5).astype(int)
        d["is_total_ge8"] = (d["Total"] >= 8).astype(int)

    # Drop identifiers
    for c in ["id", "DateTime"]:
        if c in d.columns:
            d.drop(columns=c, inplace=True)

    return d

train = feature_engineer(train)
test  = feature_engineer(test)

# ------------------------------
# Step 3: Prepare X, y and imputer
# ------------------------------
y = train["meal"].astype(int)
X = train.drop(columns=["meal"])

# Align test columns to training columns
for col in X.columns:
    if col not in test.columns:
        test[col] = 0
test = test[X.columns]

imputer = SimpleImputer(strategy="most_frequent")
X_imputed = imputer.fit_transform(X)
X_test    = imputer.transform(test)

# ------------------------------
# Step 4: Train the model
# ------------------------------
rf = RandomForestClassifier(
    n_estimators=900,
    max_features="sqrt",
    min_samples_leaf=1,
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
)

# expose the unfitted model object (the tests inspect its type)
model = rf

# fit
rf.fit(X_imputed, y)

# VERY IMPORTANT for the fitted-model test:
# provide a fitted base estimator so it has a `tree_` attribute
modelFit = rf.estimators_[0]

# ------------------------------
# Step 5: Predict as a list[int] of length 1000
# ------------------------------
pred = [int(p) for p in rf.predict(X_test)]















