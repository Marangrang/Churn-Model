import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# importing Multi-Output Classifier since we have two target variables
from sklearn.multioutput import MultiOutputClassifier
# importing GridSearch CV to find optimal set of hyperparameters
from sklearn.model_selection import GridSearchCV
# importing evaluation metrics of classification model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score,f1_score


# Load preprocessed data
#df_segment = pd.read_csv('processed_df_segment.csv')

# assigning variables for independent and dependent feature variables
#X = df_segment.drop(['Segment_Label', 'Churn'], axis= 1) # independent feature variables
#y = df_segment[['Segment_Label', 'Churn']] # dependent feature variables

#std = StandardScaler()
#X = std.fit_transform(X)

# Splitting data into train and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 2)

# Load test data
with open('X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
    
with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

# Logistic Regression
with open('logistic_regression_model.pkl', 'rb') as f:
    logreg_model = pickle.load(f)

# predicting target variables using test data with logistic regression model
y_logreg_predict = logreg_model.predict(X_test)


# Evaluation for Logistic Regression
# getting the evaluation metrics score of logistic regression model for Segmentation
print('Logistic Regression Model: Segmentation \n')

seg_report_logreg = classification_report(y_test['Segment_Label'], y_logreg_predict[:,0]) # Classification Report
print(f'Classification Report:\n {seg_report_logreg}')

seg_CM_logreg = confusion_matrix(y_test['Segment_Label'], y_logreg_predict[:,0]) # Confusion Matrix
print(f'Confusion Matrix:\n {seg_CM_logreg}\n')

seg_AS_log = round(accuracy_score(y_test['Segment_Label'], y_logreg_predict[:,0])*100,2) # Accuracy Score
seg_PS_log = round(precision_score(y_test['Segment_Label'], y_logreg_predict[:,0], average="weighted")*100,2) # Precision Score
seg_RS_log = round(recall_score(y_test['Segment_Label'], y_logreg_predict[:,0], average="weighted")*100,2) # Recall Score
seg_F1_log = round(f1_score(y_test['Segment_Label'], y_logreg_predict[:,0], average="weighted")*100,2) # F1 Score

print(f'Accuracy Sore: {seg_AS_log}')
print(f'Precision Score: {seg_PS_log}')
print(f'Recall Score: {seg_RS_log}')
print(f'F1 Score: {seg_F1_log}')

# getting the evaluation metrics score of logistic regression model for Churn Prediction
print('Logistic Regression Model: Churn Prediction \n')

churn_report_logreg = classification_report(y_test['Churn'], y_logreg_predict[:,1]) # Classification Report
print(f'Classification Report:\n {churn_report_logreg}')

churn_CM_logreg = confusion_matrix(y_test['Churn'], y_logreg_predict[:,1]) # Confusion Matrix
print(f'Confusion Matrix:\n {churn_CM_logreg}\n')

churn_AS_log = round(accuracy_score(y_test['Churn'], y_logreg_predict[:,1])*100,2) # Accuracy Score
churn_PS_log = round(precision_score(y_test['Churn'], y_logreg_predict[:,1], average="weighted")*100,2) # Precision Score
churn_RS_log = round(recall_score(y_test['Churn'], y_logreg_predict[:,1], average="weighted")*100,2) # Recall Score
churn_F1_log = round(f1_score(y_test['Churn'], y_logreg_predict[:,1], average="weighted")*100,2) # F1 Score

print(f'Accuracy Sore: {churn_AS_log}')
print(f'Precision Score: {churn_PS_log}')
print(f'Recall Score: {churn_RS_log}')
print(f'F1 Score: {churn_F1_log}')



# Random Forest Classifier
with open('random_forest_classifier_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)    

# predicting target variables using test data with random forest classifier model
y_rf_predict = rf_model.predict(X_test)

# getting the evaluation metrics score of random forest classifier model for Segmentation
print('Random Forest Classifier Model: Segmentation \n')

seg_report_rf = classification_report(y_test['Segment_Label'], y_rf_predict[:,0]) # Classification Report
print(f'Classification Report:\n {seg_report_rf}')

seg_CM_rf = confusion_matrix(y_test['Segment_Label'], y_rf_predict[:,0]) # Confusion Matrix
print(f'Confusion Matrix:\n {seg_CM_rf}\n')

seg_AS_rf = round(accuracy_score(y_test['Segment_Label'], y_rf_predict[:,0])*100,2) # Accuracy Score
seg_PS_rf = round(precision_score(y_test['Segment_Label'], y_rf_predict[:,0], average="weighted")*100,2) # Precision Score
seg_RS_rf = round(recall_score(y_test['Segment_Label'], y_rf_predict[:,0], average="weighted")*100,2) # Recall Score
seg_F1_rf = round(f1_score(y_test['Segment_Label'], y_rf_predict[:,0], average="weighted")*100,2) # F1 Score

print(f'Accuracy Sore: {seg_AS_rf}')
print(f'Precision Score: {seg_PS_rf}')
print(f'Recall Score: {seg_RS_rf}')
print(f'F1 Score: {seg_F1_rf}')

# getting the evaluation metrics score of random forest classifier model for Churn Prediction
print('Random Forest Classifier Model: Churn Prediction \n')

churn_report_rf = classification_report(y_test['Churn'], y_rf_predict[:,1]) # Classification Report
print(f'Classification Report:\n {churn_report_rf}')

churn_CM_rf = confusion_matrix(y_test['Churn'], y_rf_predict[:,1]) # Confusion Matrix
print(f'Confusion Matrix:\n {churn_CM_rf}\n')

churn_AS_rf = round(accuracy_score(y_test['Churn'], y_rf_predict[:,1])*100,2) # Accuracy Score
churn_PS_rf = round(precision_score(y_test['Churn'], y_rf_predict[:,1], average="weighted")*100,2) # Precision Score
churn_RS_rf = round(recall_score(y_test['Churn'], y_rf_predict[:,1], average="weighted")*100,2) # Recall Score
churn_F1_rf = round(f1_score(y_test['Churn'], y_rf_predict[:,1], average="weighted")*100,2) # F1 Score

print(f'Accuracy Sore: {churn_AS_rf}')
print(f'Precision Score: {churn_PS_rf}')
print(f'Recall Score: {churn_RS_rf}')
print(f'F1 Score: {churn_F1_rf}')


# KNN Classifier
with open('KNN_classifier_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

# predicting target variables using test data with KNN classifier model
y_knn_predict = knn_model.predict(X_test)

# getting the evaluation metrics score of KNN classifier model for Segmentation
print('K Nearest Neighbor Classifier Model: Segmentation \n')

seg_report_knn = classification_report(y_test['Segment_Label'], y_knn_predict[:,0]) # Classification Report
print(f'Classification Report:\n {seg_report_knn}')

seg_CM_knn = confusion_matrix(y_test['Segment_Label'], y_knn_predict[:,0]) # Confusion Matrix
print(f'Confusion Matrix:\n {seg_CM_knn}\n')

seg_AS_knn = round(accuracy_score(y_test['Segment_Label'], y_knn_predict[:,0])*100,2) # Accuracy Score
seg_PS_knn = round(precision_score(y_test['Segment_Label'], y_knn_predict[:,0], average="weighted")*100,2) # Precision Score
seg_RS_knn = round(recall_score(y_test['Segment_Label'], y_knn_predict[:,0], average="weighted")*100,2) # Recall Score
seg_F1_knn = round(f1_score(y_test['Segment_Label'], y_knn_predict[:,0], average="weighted")*100,2) # F1 Score

print(f'Accuracy Sore: {seg_AS_knn}')
print(f'Precision Score: {seg_PS_knn}')
print(f'Recall Score: {seg_RS_knn}')
print(f'F1 Score: {seg_F1_knn}')

# getting the evaluation metrics score of KNN classifier model for Churn Prediction
print('K Nearest Neighbor Classifier Model: Churn Prediction \n')

churn_report_knn = classification_report(y_test['Churn'], y_knn_predict[:,1]) # Classification Report
print(f'Classification Report:\n {churn_report_knn}')

churn_CM_knn = confusion_matrix(y_test['Churn'], y_knn_predict[:,1]) # Confusion Matrix
print(f'Confusion Matrix:\n {churn_CM_knn}\n')

churn_AS_knn = round(accuracy_score(y_test['Churn'], y_knn_predict[:,1])*100,2) # Accuracy Score
churn_PS_knn = round(precision_score(y_test['Churn'], y_knn_predict[:,1], average="weighted")*100,2) # Precision Score
churn_RS_knn = round(recall_score(y_test['Churn'], y_knn_predict[:,1], average="weighted")*100,2) # Recall Score
churn_F1_knn = round(f1_score(y_test['Churn'], y_knn_predict[:,1], average="weighted")*100,2) # F1 Score

print(f'Accuracy Sore: {churn_AS_knn}')
print(f'Precision Score: {churn_PS_knn}')
print(f'Recall Score: {churn_RS_knn}')
print(f'F1 Score: {churn_F1_knn}')


# creating a table with segmentation evaluation metrics score of different machine learning models
seg_metrics_dict = {'Models': ['Logistic Reggression', 'Random Forest Classifier', 'K Nearest Neighbor Classifier'],
               'Seg_Accuracy Score': [seg_AS_log, seg_AS_rf, seg_AS_knn],
               'Seg_Precision Score': [seg_PS_log, seg_PS_rf, seg_PS_knn],
               'Seg_Recall Score': [seg_RS_log, seg_RS_rf, seg_RS_knn],
               'Seg_F1 Score': [seg_F1_log, seg_F1_rf, seg_F1_knn]}

# creating a table with churn prediction evaluation metrics score of different machine learning models
churn_metrics_dict = {'Models': ['Logistic Reggression', 'Random Forest Classifier', 'K Nearest Neighbor Classifier'],
               'Churn_Accuracy Score': [churn_AS_log, churn_AS_rf, churn_AS_knn],
               'Churn_Precision Score': [churn_PS_log, churn_PS_rf, churn_PS_knn],
               'Churn_Recall Score': [churn_RS_log, churn_RS_rf, churn_RS_knn],
               'Churn_F1 Score': [churn_F1_log, churn_F1_rf, churn_F1_knn]}

# Save segmentation data if needed for later use
# Convert to DataFrame
seg_metrics_df = pd.DataFrame(seg_metrics_dict)
seg_metrics_df.to_csv('seg_metrics_df.csv', index=False)
print(seg_metrics_df)

# Save churn data if needed for later use
# Convert to DataFrame
churn_metrics_df = pd.DataFrame(churn_metrics_dict)
churn_metrics_df.to_csv('churn_metrics_df.csv', index=False)
print(churn_metrics_df)
