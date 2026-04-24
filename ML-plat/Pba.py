import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt

np.random.seed(42)
n = 5000

df = pd.DataFrame({
   'team_rebounds': np.random.randint(30,55,n),
   'team_assists': np.random.randint(15,35,n),
   'team_turnovers': np.random.randint(8,20,n),
   'home_game': np.random.randint(0,2,n)})

df['team_pts'] = (df['team_rebounds']*0.5+df['team_assists']*1.2-df['team_turnovers']*1.2+df['home_game']*3+np.random.randint(60, 80,n)).astype(int)
df['opp_pts']=np.random.randint(80,120,n)

df['win'] = (df['team_pts'] > df['opp_pts']).astype(int)

X = df.drop(['win', 'team_pts', 'opp_pts'],axis=1)
y = df['win']

X_train, X_test, y_train, y_test = train_test_split(
   X, y,
   test_size=0.2,
   random_state=42,
   stratify=y 
)

model = RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_split=8, min_samples_leaf=5,class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
cm = confusion_matrix(y_test, y_pred)
dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
print(X.shape)
print(y.shape)
print(X_train.shape)
print(X_test.shape)
print(f"Scores: {scores}")
print(f"Mean: {scores.mean():.3f}")
print(f"Std: {scores.std():.3f}")
print(classification_report(y_test, y_pred))
print(cm)
print("test Accuracy:", model.score(X_test, y_test))
print(df['win'].value_counts()) 
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh')
plt.title("Feature Importance")
print(df[['team_pts', 'opp_pts', 'win']].head(20))
importances = model.feature_importances_
for i, imp in enumerate(importances):
   print(f"feature {i}: {imp:.3f}")
print(f"dummy accuracy: {dummy.score(X_test, y_test):.3f}")
print(f"model score: {model.score(X_test, y_test):.3f}")
plt.show()
