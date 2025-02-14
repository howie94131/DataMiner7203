import pandas as pd
import numpy as np
from preprocessing import min_max_normalize
from joblib import load
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import cross_val_score

# Load the best model
best_model = load("model/best_model.joblib")
print("Best model loaded successfully.")

# Load the test data
test_data = pd.read_csv("raw/test_data.csv")
test_data = min_max_normalize(test_data)
print("Test data loaded successfully.")

# Make predictions on the test data
predictions = best_model.predict(test_data)

# Load training data for cross-validation (assuming preprocessed training data is available)
train_data = pd.read_csv('processed/data.csv')
X_train = train_data.drop(columns=['Target (Col 106)'])
y_train = train_data['Target (Col 106)']

# Calculate accuracy and F1-score using cross-validation
accuracy_scores = cross_val_score(best_model, X_train, y_train, scoring='accuracy', cv=5)
f1_scores = cross_val_score(best_model, X_train, y_train, scoring=make_scorer(f1_score), cv=5)
accuracy = round(np.mean(accuracy_scores), 3)
f1 = round(np.mean(f1_scores), 3)

# Prepare results for the output CSV
# Row 1 to 817: predictions, Row 818: accuracy and F1-score
results = list(predictions.astype(int))  # Convert to integer list for binary predictions
results.append(f"{accuracy},{f1},")

# Save results to CSV file with the specified format
output_path = "result/s4795041.infs4203"
with open(output_path, 'w') as f:
    # Write each prediction on a new line with trailing comma
    for pred in results[:-1]:  # For the first 817 predictions
        f.write(f"{pred},\n")
    # Write accuracy and F1-score on the last line
    f.write(f"{results[-1]}\n")

print(f"Results saved to {output_path}.")
