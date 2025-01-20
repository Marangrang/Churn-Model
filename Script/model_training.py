import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Load preprocessed data
df_segment = pd.read_csv('processed_df_segment.csv')

# assigning variables for independent and dependent feature variables
X = df_segment.drop(['Segment_Label', 'Churn'], axis= 1) # independent feature variables
y = df_segment[['Segment_Label', 'Churn']] # dependent feature variables

std = StandardScaler()
X = std.fit_transform(X)

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 2)

# Save test data for Model Evaluation
with open('X_test.pkl','wb') as f:
    pickle.dump(X_test, f)

with open('y_test.pkl','wb') as f:
    pickle.dump(y_test, f)

# importing Logistic Regression
#logreg = MultiOutputClassifier(LogisticRegression(solver='saga'))

# Pipeline for scaling and classification
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', MultiOutputClassifier(LogisticRegression(solver='saga', max_iter=5000)))
])

# setting hyperparameters of logistic regression
logreg_params = {
    'classifier__estimator__C': [0.01, 0.1, 1.0, 10.0],
    'classifier__estimator__penalty': ['l1', 'l2']
}

# using gridsearch cv to find optimal set of hyperparameters of logistic regression
logreg_gsearch = GridSearchCV(estimator= pipeline, param_grid= logreg_params, cv= 5, error_score='raise')

# fitting with training data
logreg_gsearch.fit(X_train,y_train)

# building the logistic regression model with best estimators
logreg_model = logreg_gsearch.best_estimator_



# Random Forest Model for Segmentation
rf = MultiOutputClassifier(RandomForestClassifier())

# setting hyperparameters of random forest classifier
rf_params = {
    'estimator__criterion':['gini','entropy'],
    'estimator__n_estimators':[100,200],
    'estimator__max_depth':[None,5,10],
    'estimator__max_features':[None,'sqrt','log2'],
}

# using gridsearch cv to find optimal set of hyperparameters of random forest classifier
rf_gsearch = GridSearchCV(estimator= rf, param_grid= rf_params, cv= 5)

# fitting with training data
rf_gsearch.fit(X_train,y_train)

# building the random forest classifier model with best estimators
rf_model = rf_gsearch.best_estimator_



# KNN Model for Segmentation
knn = MultiOutputClassifier(KNeighborsClassifier())

# setting hyperparameters of KNN Classifier
knn_params = {
    'estimator__n_neighbors': [3, 5, 7],
    'estimator__weights': ['uniform', 'distance'],
    'estimator__metric': ['euclidean', 'manhattan']
}

# using gridsearch cv to find optimal set of hyperparameters of KNN classifier
knn_gsearch = GridSearchCV(estimator= knn, param_grid= knn_params, cv= 5)

# fitting with training data
knn_gsearch.fit(X_train, y_train)

# building the KNN classifier model with best estimators
knn_model = knn_gsearch.best_estimator_



# Save model

with open('logistic_regression_model.pkl','wb') as f:
    pickle.dump(logreg_model, f)

#with open('random_forest_classifier_model.pkl','wb') as f:
#    pickle.dump(rf_model, f)

with open('KNN_classifier_model.pkl','wb') as f:
    pickle.dump(knn_model, f)

