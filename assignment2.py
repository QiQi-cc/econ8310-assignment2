# I choose random forest
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


# Step 1: Load data 
TRAIN_LOCAL = Path("assignment2train.csv")
TEST_LOCAL  = Path("assignment2test.csv")

TRAIN_URL = (
    "https://raw.githubusercontent.com/dustywhite7/econ8310-assignment2/main/assignment2train.csv"
)
TEST_URL = (
    "https://raw.githubusercontent.com/dustywhite7/econ8310-assignment2/main/assignment2test.csv"
)

def read_csv_safely(local_path: Path, url: str) -> pd.DataFrame:
    if local_path.exists():
        return pd.read_csv(local_path)
    else:
        print(f" Local file {local_path} not found, using GitHub version.")
        return pd.read_csv(url)

train = read_csv_safely(TRAIN_LOCAL, TRAIN_URL)
test  = read_csv_safely(TEST_LOCAL,  TEST_URL)


# Step 2: Feature engineering
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [c.strip() for c in d.columns]  

    # Parse DateTime into useful calendar features
    if "DateTime" in d.columns:
        dt = pd.to_datetime(d["DateTime"], errors="coerce")
        d["hour"]  = dt.dt.hour
        d["dow"]   = dt.dt.dayofweek
        d["month"] = dt.dt.month

        # time-of-day flags 
        d["is_breakfast"]  = d["hour"].between(6, 10).astype(int)
        d["is_lunch"]      = d["hour"].between(11, 14).astype(int)
        d["is_afternoon"]  = d["hour"].between(15, 17).astype(int)
        d["is_dinner"]     = d["hour"].between(18, 21).astype(int)
        d["is_weekend"]    = d["dow"].isin([5, 6]).astype(int)

    # Robust numeric cleaning for money columns
    for col in ["Total", "Discounts"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    # Simple caps to reduce outlier influence
    if "Total" in d.columns:
        d["Total_capped"]  = d["Total"].clip(lower=0, upper=d["Total"].quantile(0.99))
        d["is_total_gt5"]  = (d["Total"] >= 5).astype(int)
        d["is_total_gt8"]  = (d["Total"] >= 8).astype(int)
    if "Discounts" in d.columns:
        d["Discounts_capped"] = d["Discounts"].clip(
            lower=0, upper=d["Discounts"].quantile(0.99)
        )

    for c in ["id", "DateTime"]:
        if c in d.columns:
            d.drop(columns=c, inplace=True)

    return d

train = feature_engineer(train)
test  = feature_engineer(test)


# Step 3: Split X, y and impute
y = train["meal"].astype(int)
X = train.drop(columns=["meal"])

imputer = SimpleImputer(strategy="most_frequent")
X_imputed = imputer.fit_transform(X)


# Step 4: Define model, fit it, and EXPOSE THE SAME INSTANCE as modelFit
model = RandomForestClassifier(
    n_estimators=900,
    max_features="sqrt",
    min_samples_leaf=1,
    bootstrap=True,
    class_weight=None,
    random_state=42,
    n_jobs=-1,
)


model.fit(X_imputed, y)
modelFit = model  


# Step 5: Align test columns, impute, predict
for col in X.columns:
    if col not in test.columns:
        test[col] = 0
test = test[X.columns]

X_test = imputer.transform(test)
pred = [int(x) for x in modelFit.predict(X_test)]





