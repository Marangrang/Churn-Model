# Machine Learning Models for Segmentation and Churn Prediction

This repository contains a set of machine learning models used for Segmentation and Churn Prediction tasks. Three machine learning algorithms—Logistic Regression, Random Forest Classifier, and K Nearest Neighbor Classifier (KNN)—are applied and evaluated for their performance on these tasks. The evaluation metrics used for comparison include Accuracy, Precision, Recall, and F1-Score.

## Overview
#### Segmentation Task
The Segmentation task aims to group customers into different segments based on their behaviors, preferences, or purchasing patterns. The process uses RFM (Recency, Frequency, Monetary) analysis combined with Clustering techniques to segment the customer base effectively. The RFM model evaluates customer value based on:

- Recency (R): How recently a customer has made a purchase.
- Frequency (F): How often a customer makes a purchase.
- Monetary (M): How much money a customer spends.

By applying clustering algorithms (like K-Means), customers are grouped into distinct segments based on these metrics, which helps businesses understand different customer types for targeted marketing and personalized offerings.

#### Churn Prediction Task
The Churn Prediction task involves predicting whether a customer will leave (churn) or remain with the company. The model is built using historical customer data, including behaviors and transactions, with the goal of identifying which customers are at risk of churning. To create the churn label, the following approach was used:

(1) Data Splitting for Churn Label Creation:
  - The data was split into two time periods: a training period and a labeling period.
  - For each customer, their behavior during the training period was observed, and the churn label was created based on their actions in the labeling period.
    - If the customer did not make any transactions during the labeling period, they were labeled as churned (1).
    - If the customer made transactions in the labeling period, they were labeled as not churned (0).
  - This allows the model to predict churn for future customers based on past behaviors.

### Machine Learning Models Used
Three popular machine learning algorithms were employed to perform classification for both the Segmentation and Churn Prediction tasks:

- Logistic Regression: A linear classifier used for both binary and multi-class classification tasks.
- Random Forest Classifier: An ensemble method using multiple decision trees to make predictions.
- K Nearest Neighbor Classifier (KNN): A non-parametric algorithm that classifies based on the majority vote of neighboring data points.

### Evaluation Metrics
The performance of each model is evaluated using the following metrics:

- Accuracy: The proportion of correctly predicted instances.
- Precision: The proportion of true positive predictions among all positive predictions.
- Recall: The proportion of actual positive instances that were correctly predicted.
- F1-Score: The harmonic mean of Precision and Recall, offering a balance between the two.

### Results
#### Segmentation Task Evaluation:

| Model                         | Seg_Accuracy Score | Seg_Precision Score | Seg_Recall Score | Seg_F1 Score |
|-------------------------------|--------------------|---------------------|------------------|--------------|
| Logistic Regression           | 91.85             | 91.86              | 91.85           | 91.84        |
| Random Forest Classifier      | 99.90             | 99.90              | 99.90           | 99.90        |
| K Nearest Neighbor Classifier | 98.27             | 98.28              | 98.27           | 98.28        |


#### Churn Prediction Task Evaluation:

| Model                         | Churn_Accuracy Score | Churn_Precision Score | Churn_Recall Score | Churn_F1 Score |
|-------------------------------|----------------------|-----------------------|--------------------|----------------|
| Logistic Regression           | 99.84               | 99.85                | 99.84              | 99.84           |
| Random Forest Classifier       | 100.00              | 100.00               | 100.00             | 100.00          |
| K Nearest Neighbor Classifier  | 99.63               | 99.63                | 99.63              | 99.63           |


### Installation
#### Prerequisites
Ensure you have the following installed:

- Python 3.x
- pip (Python package manager)

#### Required Libraries
The following libraries are used in this project:

- scikit-learn (for machine learning algorithms)
- pandas (for data manipulation)
- numpy (for numerical computations)
- matplotlib (for visualizations, if needed)

To install the required libraries, run:

pip install -r requirements.txt

#### Dataset
The dataset used for both segmentation and churn prediction tasks must be loaded into the environment before running the scripts. Ensure that the dataset is in the correct format and properly preprocessed before training.

### How to Run
#### 1. Train the Models
To train the models for both segmentation and churn prediction tasks, execute the following commands:


python train_segmentation.py
python train_churn_prediction.py

#### 2. Evaluate the Models
After training, the models will be evaluated based on the pre-defined metrics (accuracy, precision, recall, and F1-score). The results will be printed in the console.

python evaluate_models.py

### Conclusion
- Random Forest Classifier performs the best in both Segmentation and Churn Prediction, achieving near-perfect results across all metrics.

- KNN and Logistic Regression also perform well, but Random Forest stands out as the best model for these tasks.




