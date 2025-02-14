# DataMiner
## 1. Final Choices

### Pre-processing Methods:

- Missing Value Imputation: Class-specific imputation was applied, with numerical values filled by 
class-specific means and nominal values filled by class-specific modes.
- Outlier Detection: Distance-based outlier detection using k-nearest neighbors. Rows with identified 
outliers were removed to prevent skewing of trained model.
- Normalization: Min-Max normalization was applied to scale numerical features between 0 and 1.

### Classification Model and Hyperparameters:

- K-Nearest Neighbors (K-NN):
Hyperparameters: Best k value and distance metric were selected via grid search (5-fold CV), exploring k 
values from 3 to 15 and metrics: Euclidean and Manhattan.
- Naive Bayes:
Model: Gaussian Naive Bayes, chosen for handling continuous numerical features.
Class priors were adjusted to account for class imbalance.
- Decision Tree:
Hyperparameters: max_depth, min_samples_split, and splitting criterion (Gini index, entropy).
Optimal values selected via grid search (5-fold CV).
- Random Forest:
Hyperparameters: n_estimators, max_depth, and max_features.
Best configuration determined via grid search (5-fold CV).
- Ensemble Model:
Method: Voting Classifier with hard voting, combining K-NN, Naive Bayes, Decision Tree, and Random Forest.
Final predictions were based on majority voting, and this model was selected as the final classifier for deployment.

The final decision selected K-NN as the best-performing model that is saved for usage for result prediction.

## 2. Environment Description

- Operating System: Windows 10
- Programming Language: Python 3.10

Required Packages:

- numpy (version 1.21+)
- pandas (version 1.3+)
- scikit-learn (version 0.24+)
- joblib (version 1.0+)

Install the required packages by running the following line in terminal:

`pip install numpy pandas scikit-learn joblib`


## 3. Reproduction Instructions

To execute the entire workflow, run main.py from your file path in terminal:

`python main.py`

To reproduce the results steps by steps, follow these steps:

- Data Preparation:
Ensure DM_project_24.csv and test_data.csv are placed in the raw directory, names unchanged.
The preprocessing script will load, impute, detect outliers, and normalise the data. 

`python preprocessing.py raw/rawData.csv`

The preprocessed data will be saved to processed/data.csv.

- Training Models:

Run the training script to train classifiers, perform cross-validation, and save the best model.

`python training.py`

The best model will be saved to model/best_model.joblib.

- Prediction of the Model:

Use the application script to make predictions on the test data and save the results:

`python prediction.py`

Predictions will be output to result/s4795041.csv, formatted as specified in the specsheet.

## 4. Additional Justifications
**F1-Score Selection**: F1-score was chosen over accuracy to address potential class imbalance, providing a balanced 
evaluation of model precision and recall.

**Cross-Validation**: CV was used in hyperparameter tuning and final evaluation to reduce overfitting risk, ensuring 
that models generalise well to unseen data.

**Model Choices**: The project uses 5-fold CV to compare the performance of each selected classifier method using

**F1 score**. The CV result have K-NN being the best performing model with a F1 score of 0.845. K-NN is thus chosen
to be the classifier to execute the prediction on test_data.csv. Other model such as random forest and ensemble method
also demonstrate a strong performance with a f1 score > 0.8, but K-NN is chosen due to its performance and efficiency.
