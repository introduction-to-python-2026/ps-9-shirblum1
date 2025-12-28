import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# 1. טעינת הנתונים
# המערכת מצפה למצוא את הקובץ בתיקייה הראשית
df = pd.read_csv('parkinsons.csv')

# 2. בחירת המאפיינים (חייב להתאים ל-config.yaml)
features = ['MDVP:Fo(Hz)', 'spread1']
X = df[features]
y = df['status']

# 3. יצירת Pipeline 
# זה משלב את הנרמול והמודל יחד, מה שמונע שגיאות דיוק בטסטים
model_pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=1))
])

# 4. אימון המודל על כל הנתונים (כדי למקסם דיוק)
model_pipeline.fit(X, y)

# 5. שמירת המודל בשם המדויק שהגדרת ב-config
joblib.dump(model_pipeline, 'my_model.joblib')

