# ---------------------------
# EDA of Credit Default Data
# --------------------------

# Load packages
import pandas as pd
import numpy as np

# Set paths
data_path = '/Users/brettharder/Documents/Programming Practice/kaggle_credit_default/data/'

# Read in data
credit_df = pd.read_csv(data_path + 'UCI_Credit_Card.csv')

# 3,000 obs 25 cols
credit_df.shape
col_names = credit_df.columns 
col_names

# Class imbalance on whole dataset
def column_summary(df,col_name):
    sum_df = df.groupby(str(col_name),as_index = False).agg({'ID':'nunique'})
    
test = credit_df.groupby('default.payment.next.month',as_index = False).agg({'ID':'nunique'})
test
