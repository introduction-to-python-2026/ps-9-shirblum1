import pandas as pd
import yaml
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# 1. Load config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

features = config["features"]        # list of 2 features
model_path = config["path"]          # where to save the model


# 2. Load dataset
df = pd.read_csv("parkinsons.csv")

X = df[features]
y = df["status"]


# 3. Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 5. Train model
model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train, y_train)


# 6. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")


# 7. Save model
joblib.dump(model, model_path)

