import pandas as pd
# טעינת הקובץ כ-DataFrame
df = pd.read_csv('parkinsons.csv')
import joblib
joblib.dump(model, 'my_model.joblib')
# שינוי המאפיינים כדי להעלות את הדיוק
features = ['MDVP:Fo(Hz)', 'spread1']
X = df[features]
y = df['status']
print(f"נבחרו מאפיינים חדשים: {features}")
from sklearn.preprocessing import MinMaxScaler
# נרמול הנתונים לטווח של 0 עד 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
# פיצול לסט אימון וסט בדיקה
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
# ננסה שכן אחד קרוב ביותר
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
# בדיקת הדיוק על סט הבדיקה
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


