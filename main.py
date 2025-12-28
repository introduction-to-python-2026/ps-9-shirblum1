import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# 1. טעינת הנתונים
df = pd.read_csv('parkinsons.csv')

# 2. בחירת המאפיינים (חייב להתאים בדיוק ל-config.yaml)
features = ['MDVP:Fo(Hz)', 'spread1']
X = df[features]
y = df['status']

# 3. יצירת Pipeline שכולל גם את הנרמול וגם את המודל
# זה מבטיח שהמודל ינרמל נתונים חדשים באותו אופן בדיוק
model_pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=1))
])

# 4. אימון ה-Pipeline כולו
model_pipeline.fit(X, y)

# 5. שמירת ה-Pipeline כמודל הסופי
joblib.dump(model_pipeline, 'my_model.joblib')
