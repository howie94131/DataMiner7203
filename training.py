import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import make_scorer, f1_score
from joblib import dump

# Load preprocessed data

data = pd.read_csv('processed/data.csv')
X = data.drop(columns=['Target (Col 106)'])
y = data['Target (Col 106)']

# Define F1-score as scoring metric
f1_scorer = make_scorer(f1_score)

### 1. K-Nearest Neighbors
knn_params = {
    'n_neighbors': range(3, 16),
    'metric': ['euclidean', 'manhattan']
}
knn = GridSearchCV(KNeighborsClassifier(), knn_params, scoring=f1_scorer, cv=5)
knn.fit(X, y)
best_knn = knn.best_estimator_
print("Best KNN Model:", knn.best_params_)
print("KNN F1-Score:", cross_val_score(best_knn, X, y, scoring=f1_scorer, cv=5).mean())

### 2. Naive Bayes
nb_params = {
    'priors': [None]  # Could add class priors if required
}
nb = GridSearchCV(GaussianNB(), nb_params, scoring=f1_scorer, cv=5)
nb.fit(X, y)
best_nb = nb.best_estimator_
print("Naive Bayes F1-Score:", cross_val_score(best_nb, X, y, scoring=f1_scorer, cv=5).mean())

### 3. Decision Tree
dt_params = {
    'max_depth': range(3, 21),
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 5, 10]
}
dt = GridSearchCV(DecisionTreeClassifier(), dt_params, scoring=f1_scorer, cv=5)
dt.fit(X, y)
best_dt = dt.best_estimator_
print("Best Decision Tree Model:", dt.best_params_)
print("Decision Tree F1-Score:", cross_val_score(best_dt, X, y, scoring=f1_scorer, cv=5).mean())

### 4. Random Forest
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'max_features': ['sqrt', 'log2']
}
rf = GridSearchCV(RandomForestClassifier(), rf_params, scoring=f1_scorer, cv=5)
rf.fit(X, y)
best_rf = rf.best_estimator_
print("Best Random Forest Model:", rf.best_params_)
print("Random Forest F1-Score:", cross_val_score(best_rf, X, y, scoring=f1_scorer, cv=5).mean())

### 5. Ensemble Model - Voting Classifier
ensemble = VotingClassifier(
    estimators=[
        ('knn', best_knn),
        ('nb', best_nb),
        ('dt', best_dt),
        ('rf', best_rf)
    ],
    voting='hard'
)
ensemble_score = cross_val_score(ensemble, X, y, scoring=f1_scorer, cv=5).mean()
print("Ensemble F1-Score:", ensemble_score)

# Compare model scores and select the best
models = {
    'KNN': best_knn,
    'Naive Bayes': best_nb,
    'Decision Tree': best_dt,
    'Random Forest': best_rf,
    'Ensemble': ensemble
}
f1_scores = {
    model_name: cross_val_score(model, X, y, scoring=f1_scorer, cv=5).mean()
    for model_name, model in models.items()
}
best_model_name = max(f1_scores, key=f1_scores.get)
best_model = models[best_model_name]

print(f"Best model is {best_model_name} with F1-Score: {f1_scores[best_model_name]}")

# Save the best model
output_path = "model/best_model.joblib"
dump(best_model, output_path)
print(f"Best model saved to {output_path}.")
