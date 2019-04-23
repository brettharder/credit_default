# --------------------------------
# Credit Default Prelim XGBOOST!!
# -------------------------------

# Load packages
import pandas as pd
import numpy as np
import os
import time

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict,cross_val_score,train_test_split,StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc,precision_recall_curve,roc_curve, roc_auc_score

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
%matplotlib inline

# Set paths
data_path = '/Users/brettharder/Documents/Programming Practice/kaggle_credit_default/data/'

# Read in data
credit_df = pd.read_csv(data_path + 'UCI_Credit_Card.csv')

# 3,000 obs 25 cols
credit_df.shape

# --------------------------------------
# Clean data and create dummy variables
# --------------------------------------

# Column names
col_names = credit_df.columns

"""
'ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
       'default.payment.next.month'

1. Drop ID
2. Dummy
    - SEX
    - EDUCATION
    - MARRIAGE
    - PAY_0 -> PAY_6
    (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months,
    8=payment delay for eight months, 9=payment delay for nine months and above)
3. Strip off y value -> default.payment.next.month       
"""

# List of features for dummies
dum_list = ['SEX','EDUCATION','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
# loop through list and create dummies!
for i in dum_list:
    dummies = pd.get_dummies(credit_df[str(i)]).rename(columns=lambda x: str(i)+'_' + str(x))
    credit_df = pd.concat([credit_df, dummies], axis=1)

# Dummies created, now drop one dummy for each category for modelling
new_cols = credit_df.columns

# Drop all _2 dummies
drop_cols = [x for x in new_cols if x.endswith('_2')]
drop_cols
credit_df.drop(drop_cols, axis=1, inplace=True)

# Now also drop original columns that were dummified as well as ID and 
new_drop = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6','ID']
credit_df.drop(new_drop, axis=1, inplace=True)

# ---------------------------------------------------------------
# Save modelling data
# ---------------------------------------------------------------

#credit_df.to_csv(data_path +'/modelling_data/modelling_data.csv',index = False)

# ---------------------------------------------------------------
# Split into test and train dataset 
# ---------------------------------------------------------------

# split data into X and y
Y = credit_df["default.payment.next.month"]
X = credit_df.drop("default.payment.next.month", axis=1,)

# split data into train and test sets
seed = 7
test_size = 0.4
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# ----------------------------------------------------------------
# Run CV on initail XGB model
# ----------------------------------------------------------------

# Create classifier (default hyperparams)
model = XGBClassifier()
# Specify CV method (stratified becasue of class imbalance)
kfold = StratifiedKFold(n_splits=10, random_state=7)
# Get results of CV
results = cross_val_score(model, X_train, y_train, cv=kfold)
# 82% Accuracy on training set w/ default hyperparams
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# -----------------------------------------------------------------
# Train model on whole train set
# -----------------------------------------------------------------

# fit model on training data
model = XGBClassifier()
model.fit(X_train, y_train)

# -----------------------------------------------------------------
# Look @ model performance on holdout set
# -----------------------------------------------------------------

# make predictions for test data
y_pred = model.predict_proba(X_test)[:,1]

# Round predictions for accuracy
predictions = [round(value) for value in y_pred]

# 82% Accuracy on holdout set (CV was good at estimating models performance!)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# -----------------------------------------------------------------
# PR Curve
# -----------------------------------------------------------------

model_predictions = pd.DataFrame({
        "true_value":y_test,
        "prediction":predictions,
        "probability":y_pred       
        })

precision, recall, _ = precision_recall_curve(y_test, y_pred)

# Plot PR Curve
fig = plt.figure(figsize=(15,8))
ax1 = fig.add_subplot(1,2,1)
ax1.set_xlim([-0.05,1.05])
ax1.set_ylim([-0.05,1.05])
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.set_title('PR Curve')
p,r,_ = precision_recall_curve(y_test,y_pred)
ax1.plot(r,p)
plt.show()

# ---------------------------
# ROC
# ---------------------------

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='XGBOOST')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# 77% AUC
roc_auc_score(y_test, y_pred)

# -----------------------------------------------------------------
# Hyperparameter tuning (Random Search)
# -----------------------------------------------------------------




