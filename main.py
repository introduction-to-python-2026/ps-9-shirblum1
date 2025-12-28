import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. טעינת הנתונים
df = pd.read_csv('parkinsons.csv')

# 2. בחירת המאפיינים (שימי לב לשמות המדויקים)
features = ['MDVP:Fo(Hz)', 'spread1']
X = df[features]
y = df['status']

# 3. נרמול
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4. פיצול הנתונים
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. מודל KNN עם שכן אחד (כפי שעשית בקולאב)
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)

# 6. שמירת המודל - השם חייב להיות זהה למה שכתוב ב-config
joblib.dump(model, 'my_model.joblib')


