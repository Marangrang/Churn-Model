import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the AKQA dataset
AKQA_data = pd.read_excel('../Data/Marketing_Agency_Segmentation_Churn_Predictive_Analytics.xlsx')

# Preprocessing: Handling missing values, encoding categorical variables if needed

# Convert 'Gender' column to numeric (2 for 'Non-Binary', 1 for 'Male', 0 for 'Female')
AKQA_data['Gender_num'] = AKQA_data['Gender'].map({'Female': 0, 'Male': 1, 'Non-Binary': 2})

# Convert 'Income_Level' column to numeric (2 for 'High', 1 for 'Medium', 0 for 'Low')
AKQA_data['Income_Level_num'] = AKQA_data['Income_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})

# Convert 'Income_Level' column to numeric (2 for 'High', 1 for 'Medium', 0 for 'Low')
AKQA_data['Income_Level_num'] = AKQA_data['Income_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})

# Splitting the whole dataset into two parts.
# first part without last 3 months of transactions

## getting the latest invoice date in the dataset
latest_invoice_date = AKQA_data['Transaction_Date'].max()
# invoice date before three months of latest date
mon3_ret_date = pd.Timestamp('2023-10-02')

# taking the first part of data that doesn't have last 3 months of transaction

df_part1 = AKQA_data.copy() # creating copy
df_part1.set_index('Transaction_Date', inplace= True) # setting Date as index
df_part1 = df_part1.loc[:'2023-10-02'] # slicing the data
# reseting the index
df_part1.reset_index(inplace= True) # restor


# Feature engineering for RFM (Recency, Frequency, Monetary) analysis
# Assuming 'last_purchase_date', 'total_transactions', and 'total_spent' columns exist
# Convert 'Transaction_Date' to datetime, coercing errors
df_part1['Transaction_Date'] = pd.to_datetime(df_part1['Transaction_Date'], errors='coerce')

# getting the latest transaction Transaction_Date in df_part1
df_part1_latest_date = df_part1['Transaction_Date'].max()

#Recency
# calculating the recency of each customer

recency = df_part1.groupby('CustomerID').agg({'Transaction_Date': lambda x :
                                              (df_part1_latest_date - x.max()).days}).reset_index() # calculating recency
recency.rename(columns= {'Transaction_Date':'Recency'}, inplace= True) # renaming columns
# Frequency
# calculating the frequency of each customer
frequency = df_part1.groupby('CustomerID').agg({'CustomerID':'count'}) # calculating frequency
frequency.rename(columns= {'CustomerID':'Frequency'}, inplace= True) # renaming columns
frequency.reset_index(inplace= True) # resetting index

# Monetary
# calculating the monetary of each customer
monetary = df_part1.groupby('CustomerID').agg({'Purchase_Amount':'sum'}).reset_index() # calculting monetary
monetary.rename(columns= {'Purchase_Amount':'Monetary'}, inplace= True) # renaming columns

# creating a new dataframe RFM with recency, frequency and monetary of each customer
RFM = pd.concat([recency.iloc[:,:], frequency.iloc[:,-1], monetary.iloc[:,-1]], axis= 1)

# sorting the recency, frequency and monetary values of customers into bins using pd.cut() to get their respective scores

RFM['R_Score'] = pd.cut(RFM['Recency'], bins= [-1, 30, 70, 140, RFM['Recency'].max()],
                        labels= [4,3,2,1]).astype('int64') # getting R_Score for each customer

RFM['F_Score'] = pd.cut(RFM['Frequency'], bins= [0, 5, 10, 15, RFM['Frequency'].max()],
                        labels= [1,2,4,8]).astype('int64') # getting R_Score for each customer

RFM['M_Score'] = pd.cut(RFM['Monetary'], bins= [0, 1000, 2500, 4500, RFM['Monetary'].max()],
                        labels= [1,3,6,10]).astype('int64') # getting R_Score for each customer

# concatenating the scores for each customer to get their values
RFM['RFM_Value'] = RFM.apply(lambda x: str(int(x['R_Score'])) + str(int(x['F_Score'])) + str(int(x['M_Score'])), axis=1)

# adding the the scores for each customer to get their overall score
RFM['RFM_Score'] = RFM['R_Score'] + RFM['F_Score'] + RFM['M_Score']

# K-Means Clustering
# assigning the required independent feature variables of RFM dataframe to X_rfm variable
X_rfm = RFM[['R_Score', 'F_Score', 'M_Score', 'RFM_Value', 'RFM_Score']]

# standardizing the data with StandardScaler
std_scaler = StandardScaler()
X_rfm = std_scaler.fit_transform(X_rfm)

# KMeans clustering for Segmentation (RFM)
# segmenting each customer into different segments based on their RFM scores

RFM['Seg_Num'] = pd.cut(RFM['RFM_Score'], bins= [0, 5, 11, 16, 20],
                               labels= [4,3,2,1]) # getting Segment_Number for each customer

RFM['Segment_Label'] = pd.cut(RFM['RFM_Score'], bins= [0, 5, 11, 16, 20],
                              labels= ['Basic Customer', 'Standard Customer',
                                       'Prime Customer', 'Elite Customer']) # getting Segment_Label for each customer

# Present Data
# taking the second part of data that have only last 3 months of transaction

df_part2 = AKQA_data.copy() # creating copy
df_part2.set_index('Transaction_Date', inplace= True) # setting Date as index
df_part2 = df_part2.loc['2023-10-02':] # slicing the data
df_part2.reset_index(inplace= True) # resetting index

# Convert 'Transaction_Date' to datetime, coercing errors
df_part2['Transaction_Date'] = pd.to_datetime(df_part2['Transaction_Date'], errors='coerce')

# getting the number of customers in part1 and part2
part1_customer = df_part1['CustomerID'].sort_values().unique()
part2_customer = df_part2['CustomerID'].sort_values().unique()

# finding how many old customers made transaction in last three months

R_next_3months = [] # empty list to store customer ID

for i in part1_customer:
    if i in part2_customer: # checking customer of part1 data in part2 data
        R_next_3months.append('Yes') # if true append Yes

    else:
        R_next_3months.append('No') # else append No

RFM['R_Next_3Months'] =  R_next_3months # adding the new feature variable

# finding whether the customer is churned or not based on conditions

Churn = [] # empty list to store the status of churn of customer

for i,j in enumerate(RFM['CustomerID']):

    if RFM['Recency'][i] <= 90 and RFM['R_Next_3Months'][i] == 'Yes':
        Churn.append('No')

    elif RFM['Recency'][i] <= 90 and RFM['R_Next_3Months'][i] == 'No':

        if RFM['Frequency'][i] <= 15:
            Churn.append('High Risk')

        else:
            Churn.append('Low Risk')

    elif RFM['Recency'][i] > 90 and RFM['R_Next_3Months'][i] == 'Yes':

        if RFM['Frequency'][i] > 15:
            Churn.append('No')

        else:
            Churn.append('Low Risk')

    elif RFM['Recency'][i] > 90 and RFM['R_Next_3Months'][i] == 'No':
        Churn.append('Yes')

RFM['Churn'] = Churn # adding the new feature variable churn

# dropping unwanted and multicollinearity feature variables
df_segment = RFM.copy() # copy of RFM dataframe
df_segment.drop(['CustomerID', 'R_Score', 'F_Score', 'M_Score', 'RFM_Score', 'RFM_Value', 'Seg_Num'], axis= 1, inplace= True) # dropping

print(df_segment.head())

# removing outliers using zscore
zscore_recency = np.abs(stats.zscore(df_segment['Recency'])) # calculating Z-score for recency
zscore_monetary = np.abs(stats.zscore(df_segment['Monetary'])) # calculating Z-score for monetary
threshold = 3 # setting threshold value
outliers_recency = list(np.where(zscore_recency>threshold)[0]) # getting outliers index in recency
outliers_monetary = list(np.where(zscore_monetary>threshold)[0]) # getting outliers index in monetary
outliers_indices = list(set(outliers_recency + outliers_monetary)) # creating a set for getting unique index of outliers
outliers_indices.sort() # sorting the list
df_segment = df_segment.drop(df_segment.index[outliers_indices]) # dropping outlier records

# encoding categorical variable using Label Encoder
labelencoder = LabelEncoder()
df_segment['R_Next_3Months'] = labelencoder.fit_transform(df_segment['R_Next_3Months'])


# Splitting the data into features (X) and target (y) for segmentation
# Encode categorical target variables as integers
df_segment['Segment_Label'] = labelencoder.fit_transform(df_segment['Segment_Label'])
df_segment['Churn'] = labelencoder.fit_transform(df_segment['Churn'])

# Save processed data if needed for later use
df_segment.to_csv('processed_df_segment.csv', index=False)

print("Preprocessing complete. Data ready for training.")
