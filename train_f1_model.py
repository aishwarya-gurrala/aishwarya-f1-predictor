import pandas as pd
from xgboost import XGBClassifier
import pickle

# Load CSV files (make sure they're saved as .csv)
results = pd.read_csv("results.csv")
races = pd.read_csv("races.csv")

# Join on raceId
df = results.merge(races, on="raceId")

# Filter rows with positionOrder
df = df[df['positionOrder'].notnull()]

# Create binary target: 1 if driver won the race, 0 otherwise
df['won'] = df['positionOrder'].apply(lambda x: 1 if x == 1 else 0)

# Select input features (feel free to modify later)
df = df.dropna(subset=['grid', 'points'])  # drop missing rows
X = df[['grid', 'points']]                 # grid = starting position, points = driver's season points
y = df['won']

# Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X, y)

# Save the model
with open("f1_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as f1_model.pkl")
