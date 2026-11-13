import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from knnImputation import knnImputation
from outlierDelete import outlierDelete

#Read Data
# The data file is not included in this repository: put TrainDataset2025.csv in the data/ folder
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'TrainDataset2025.csv')
all_df = pd.read_csv(DATA_PATH, index_col=False)
all_df.drop('ID', axis=1, inplace=True)


#Handle missing value
"""
For the regression(RFS) model, there are no missing values in the target variable (y).
For the classification(PCR) model, there are missing values in the target variable (y).
Extract the columns (of continuous numerical type) that contain missing values.
For the feature variables (x), use the mean value for data imputation.
"""
df_mean = SimpleImputer(missing_values=999, strategy='mean',copy=False)
missing_cols_continues = [c for c in all_df.columns if c not in ['pCR (outcome)','RelapseFreeSurvival (outcome)','ER','PgR','HER2','TrippleNegative','ChemoGrade','Proliferation','HistologyType','LNStatus','TumourStage','Gene']]
all_df.loc[:,missing_cols_continues]=df_mean.fit_transform(all_df.loc[:, missing_cols_continues])

#Check if all missing values in the consecutive segments of the data have been handled?
for cols in missing_cols_continues:
    if (all_df.loc[:,cols]==999).any():
        print('error,also has 999')
#print('ok,no more 999')

#Filling Missing Values for Discrete Numerical Features
#Fill Gene with KNN
knnImputation('Gene',all_df)

#Fill LNStatus with KNN
knnImputation('LNStatus',all_df)

#Fill HistologyType with KNN
knnImputation('HistologyType',all_df)

#Fill Proliferation with KNN
knnImputation('Proliferation',all_df)

#Fill ChemoGrade with KNN
knnImputation('ChemoGrade',all_df)

#Fill TrippleNegative with KNN
knnImputation('TrippleNegative',all_df)

#Fill HER2 with KNN
knnImputation('HER2',all_df)

#Fill PgR with KNN
knnImputation('PgR',all_df)

#Delete the row which has 999
all_df = all_df[all_df['pCR (outcome)'] != 999]

#Check also has 999 or not
#print('error,also has 999' if (all_df.loc[:,:]==999).any().any() else 'ok,no more 999')

#Outlier handling
"""
The number of points with a value of 1 is 84, and the number of points with a value of 0 is 311. Due to the extremely unbalanced sample, 
if a box plot is drawn as a whole to look at the outlier's plot, the data such as Q1,Q3, and IQR are largely biased towards the data with 
a value of 0, which leads to the point with a value of 1 being easily mistaken for an outlier
Split the samples and conduct outlier analysis and processing on those with values of 1 and 0 respectively
"""
all_df_1=all_df[all_df['pCR (outcome)']==1]
all_df_0=all_df[all_df['pCR (outcome)']==0]
#Handle outliers separately --> Delete outliers (points exceeding the upper and lower limits of the box diagram)
all_df_1_cleaned=outlierDelete(all_df_1)
all_df_0_cleaned=outlierDelete(all_df_0)
all_df_cleaned = pd.concat([all_df_0_cleaned, all_df_1_cleaned], axis=0) 
#0:311--->136;1:84--->30

'''
#Count 0 and 1
num_1 = (all_df_cleaned['pCR (outcome)'] == 1).sum()
num_0 = (all_df_cleaned['pCR (outcome)'] == 0).sum()
print(f"1: {num_1}, 0: {num_0}")
'''

#Numerical imbalance handling
# 30:136---> 1:4.53  data does not exhibit significant imbalances

y1=all_df_cleaned["pCR (outcome)"]
y2=all_df_cleaned["RelapseFreeSurvival (outcome)"]
X=all_df_cleaned.drop('pCR (outcome)',axis=1).drop('RelapseFreeSurvival (outcome)',axis=1)
data = all_df_cleaned.copy()
