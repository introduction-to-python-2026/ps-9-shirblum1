import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# 1. Load the dataset
df = pd.read_csv("parkinsons.csv")

# 2. Select features and target
features = ["MDVP:Fo(Hz)", "HNR"]
target = "status"

X = df[features]
y = df[target]

# 3. Scale the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Choose and train the model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# 6. Test the accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy}")

# 7. Save the model
joblib.dump(model, "parkinson_model.joblib")

