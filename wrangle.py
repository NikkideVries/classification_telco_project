# Imports:
# acquire imports
import pandas as pd
import numpy as np
import os
import env


#preperation imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
#--------------------------------------------------------#
#Variables 



#--------------------------------------------------------#
# Data Acquistion:
#--------------------------------------------------------#

def get_db_url(db, user= env.user, host=env.host, password=env.password):
    """
    This function will:
    - take credentials from env.py file
    - make a connection to the SQL database with given credentials
    - return url connection
    """
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


def new_telco_data():
    '''
    This function will:
    - read a set sql query
    - return a dataframe based on the given query
    '''
    
    telco_query = """
                select * from customers
                join contract_types using (contract_type_id)
                join internet_service_types using (internet_service_type_id)
                join payment_types using (payment_type_id)
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(telco_query, get_db_url('telco_churn'))
    
    return df

def get_telco_data():
    '''
    This function will check for a telco.csv. 
    If it exists it will pull data from said file.
    '''
    if os.path.isfile('telco.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('telco.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_telco_data()
        
        # Cache data
        df.to_csv('telco.csv')
        
    return df

#---------------------------------------------------------#
#Data Preperation:
#---------------------------------------------------------#
def clean_telco_data():
    """
    This function will:
    - drop duplicate and unessasary columns
    - encode total_charges into a float
    """
    df = get_telco_data()
    # Drop duplicate columns
    df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id'], inplace=True)
    
    #change total charges into a float
    df.loc[:,'total_charges'] = (df.total_charges + '0')
    df.total_charges = df.total_charges.astype(float)
    
    return df

def split_telco_data():
    '''
    This function will:
    - run clean_telco_data 
    - split the data into train, validate, test
    '''
    df = clean_telco_data()
    train_validate, test = train_test_split(df, test_size=0.2, random_state=1234, stratify=df.churn)
    train, validate = train_test_split(train_validate, train_size=0.7, random_state=1234, stratify=train_validate.churn)
    return train, validate, test

# Dummy columns: 






#---------------------------------------------------------#
# Explore functions: 
#---------------------------------------------------------#
# look a univarite numerical columns:
def visualize_univariate_num(train, num_cols):
    for col in num_cols:
        # Print out the title
        print(f'Distribution of {col}')
        
        # Show descriptive statistics
        print(train[col].describe())
        
        # First graph is a histogram
        sns.histplot(data=train, x=col, kde=True)
        plt.show()
        
        # Second graph is a boxplot
        sns.boxplot(data=train, x=col)
        plt.show()
        
        print('=======================')
        
#look at univarite caegorical variables:
def visualize_univariate_cat(train, cat_cols):
    for col in cat_cols:
        # Print the frequency of the categorical variable
        print(f'Frequency of {col}')
        print(train[col].describe())
        print(train[col].value_counts())
        
        # Create a side-by-side subplot with two plots
        fig, ax = plt.subplots(1, 2, figsize=(10, 8))
        fig.suptitle(f'Graphs of {col}')
        
        # Plot one: Countplot (bar chart)
        sns.countplot(data=train, x=col, ax=ax[0], palette='Set2')
        
        # Plot two: Boxplot
        sns.boxplot(data=train, x=col, y=train[col].value_counts(), ax=ax[1], color='skyblue')
        
        plt.show()
        print('----------###---------')
        
