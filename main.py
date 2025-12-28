import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. טעינת הנתונים
df = pd.read_csv('parkinsons.csv')

# 2. בחירת המאפיינים (אלו שהביאו לך מעל 0.8)
features = ['MDVP:Fo(Hz)', 'spread1'] 
X = df[features]
y = df['status']

# 3. נרמול הנתונים
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4. פיצול הנתונים
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. יצירת מודל KNN ואימונו
model = KNeighborsClassifier(n_neighbors=1) # השתמשי ב-K שהשתמשת בו ב-Notebook
model.fit(X_train, y_train)

# 6. בדיקת דיוק (להדפסה בלוגים)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Accuracy: {accuracy}")

# 7. שמירת המודל לקובץ - חובה עבור הבדיקה האוטומטית
joblib.dump(model, 'parkinson_model.joblib')


