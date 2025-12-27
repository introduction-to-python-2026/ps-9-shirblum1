import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib

# load data
df = pd.read_csv("parkinsons.csv")

# select features and target
X = df[["MDVP:Fo(Hz)", "HNR"]]
y = df["status"]

# train model
model = KNeighborsClassifier(n_neighbors=7)
model.fit(X, y)

# save model
joblib.dump(model, "parkinson_model.joblib")

