import pandas as pd
import joblib
import yaml

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

features = config["features"]
path = config["path"]

# Load dataset
df = pd.read_csv("parkinsons.csv")

X = df[features]
y = df["status"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Model pipeline
model = Pipeline([
    ("scaler", MinMaxScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

# Train
model.fit(X_train, y_train)

# Save model
joblib.dump(model, path)

