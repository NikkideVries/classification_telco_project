# Imports:
# acquire imports
import pandas as pd
import numpy as np
import os
import env

#visaulization
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split


#preperation imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# imports for modeling
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


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

# split 
def train_validate_test():
    """
    train_validate_test will take one argument(df) and 
    run clean_telco to remove/rename/encode columns
    then split our data into 20/80, 
    then split the 80% into 30/70
    
    perform a train, validate, test split
    
    return: the three split pandas dataframes-train/validate/test
    """
    df = clean_telco_model()    
    
    train_validate, test = train_test_split(df, test_size=0.2, random_state=3210, stratify=df.churn)
    train, validate = train_test_split(train_validate, train_size=0.7, random_state=3210, stratify=train_validate.churn)
    return train, validate, test

#---------------------------------------------------------#
#Data Preperation:
#---------------------------------------------------------#

def clean_telco_data(df):
    """
    This function will:
    - drop duplicate and unessasary columns
    - encode total_charges into a float
    """
    
    # Drop duplicate columns
    df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id'], inplace=True)
    
    
    df.loc[:, 'did_churn'] = (df.churn == 'Yes').astype(int)
    #change total charges into a float
    df.loc[:,'total_charges'] = (df.total_charges + '0')
    df.total_charges = df.total_charges.astype(float)
    
    return df
    
def split_telco_data(df):
    '''
    This function will:
    - run clean_telco_data 
    - split the data into train, validate, test
    '''
    df = clean_telco_data(df)
    train_validate, test = train_test_split(df, test_size=0.2, random_state=1234, stratify=df.churn)
    train, validate = train_test_split(train_validate, train_size=0.7, random_state=1234, stratify=train_validate.churn)
    return train, validate, test








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
        

        
        
#--------------------------------------------------------#
# calculations univariate
#--------------------------------------------------------#

#normal distribution: 

def eval_dist(r, p, α=0.05):
    """
    This function will take in:
    - r: the test statistic
    - p: p-value
    - α: id defaulted to 5%
    and print out if the data used to create r & p from the stats.shapiro test is normally distributed.
    """
    if p > α:
        return print(f"""The data is normally distributed""")
    else:
        return print(f"""The data is NOT normally distributed""")
    
    
def evaluate_and_print_normality(data, column_name):
    """
    This function will take in a column_name, and then run a shapiro test
    """
    r, p = stats.shapiro(data[column_name])
    print(f"{column_name} distribution:")
    print("Shapiro-Wilk Test Results:")
    print(f"Statistic (r): {r}")
    print(f"P-value (p): {p}")
    print(eval_dist(r, p))
    
    
def normal_test(train):
    """ 
    This function will:
    - run evaluate and print normalility for a list of numerical columns
    """
    
    numerical_columns = ["tenure", "monthly_charges","total_charges"]

    for column in numerical_columns:
        evaluate_and_print_normality(train, column)
        
        
#------------------------------------------------------------#
#model visualization
#-------------------------------------------------------------#
def visualize_univariate_cat_final(train):
    '''
    data visualization for final notebook of specific categorical columns. data is univariate
    '''
    cat_cols = ['senior_citizen', 'contract_type', 'internet_service_type', 'payment_type', 'churn', 'dependents']
    
    # Define the number of rows and columns for subplots
    num_cols = 2
    num_rows = (len(cat_cols) + num_cols - 1) // num_cols
    
    # Create a side-by-side subplot for each categorical variable
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 12))
    fig.suptitle('Graphs of Categorical Variables', fontsize=16)
    
    for i, col in enumerate(cat_cols):
        row = i // num_cols
        col_num = i % num_cols
        
        
        # Plot one: Countplot (bar chart)
        sns.countplot(data=train, x=col, ax=axes[row, col_num], palette='Set2')
        axes[row, col_num].set_title(f'Countplot of {col}')
        
        
        

    plt.tight_layout()
    plt.show()


def create_subplots_barplots_with_mean(data, target, num_cols=['tenure', 'monthly_charges', 'total_charges'], num_cols_per_row=2, figsize=(15, 12)):
    '''
    creates barplots for numerical data for final teclo notebook
    '''
    num_rows = (len(num_cols) + num_cols_per_row - 1) // num_cols_per_row

    fig, axes = plt.subplots(num_rows, num_cols_per_row, figsize=figsize)
    fig.suptitle('Bar Plots of Numerical Columns with Mean', fontsize=16)

    for i, col in enumerate(num_cols):
        row = i // num_cols_per_row
        col_num = i % num_cols_per_row
        ax = axes[row, col_num]

        ax.set_title(f'Graph of {col}')
        sns.barplot(x=target, y=col, data=data, palette='mako', ax=ax)
        col_mean = data[col].mean()
        ax.axhline(col_mean, label=f'Mean of {col}', color='black')
        ax.legend()

    plt.tight_layout()
    plt.show()

    

def create_subplots_barplots_categorical_vs_target(data, target, num_cols_per_row=2, figsize=(15, 12)):
    '''
    creates barplots for categorical vs target for final teclo notebook
    '''
    col_list = ['senior_citizen', 'contract_type', 'internet_service_type', 'payment_type']
    num_rows = (len(col_list) + num_cols_per_row - 1) // num_cols_per_row

    fig, axes = plt.subplots(num_rows, num_cols_per_row, figsize=figsize)
    fig.suptitle('Bar Plots of Categorical Columns vs. Target', fontsize=16)

    for i, col in enumerate(col_list):
        row = i // num_cols_per_row
        col_num = i % num_cols_per_row
        ax = axes[row, col_num]

        ax.set_title(f'{col} vs. {target}')
        sns.barplot(data=data, x=col, y=target, palette='viridis', ax=ax)
        overall_rate = data[target].mean()
        ax.axhline(overall_rate, ls='--', color='black')

    plt.tight_layout()
    plt.show()

   

#-------------------------------------------------------------#
#hypothesis functions
#-------------------------------------------------------------#


def perform_chi_squared_test(observed, null_hypothesis, alternative_hypothesis, alpha=0.05):
    '''
    preform a chi2 test and print if the null is accepted or rejected
    '''
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    if p < alpha:
        print(f"We reject the null hypothesis: {null_hypothesis}")
        print(f"Therefore: {alternative_hypothesis}")
    else:
        print(f"We fail to reject the null, therefore: {null_hypothesis}")
    print(f"P-value: {p}")

#--------------------------------------------------------------#
#modeling functions
#--------------------------------------------------------------#
def clean_telco_model():
    """
    This function will:
    - drop duplicate and unessasary columns
    - encode total_charges into a float
    """
    df = get_telco_data()
    # Drop duplicate columns
    df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id'], inplace=True)
    
    
    df.loc[:, 'did_churn'] = (df.churn == 'Yes').astype(int)
    #change total charges into a float
    df.loc[:,'total_charges'] = (df.total_charges + '0')
    df.total_charges = df.total_charges.astype(float)
    return df

def create_dummy_columns(df):
    
    dummies_list = []
    cat_cols = ['gender',
                'senior_citizen',
                'partner',
                'dependents',
                'phone_service',
                'multiple_lines',
                'online_security',
                'online_backup',
                'device_protection',
                'tech_support',
                'streaming_tv',
                'streaming_movies',
                'paperless_billing',
                'contract_type',
                'internet_service_type',
                'payment_type']
    for col in cat_cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        dummies_list.append(dummies)
    
    df = pd.concat([df] + dummies_list, axis=1)
    
    return df

# split 
def train_validate_test(df):
    """
    train_validate_test will take one argument(df) and 
    run clean_telco to remove/rename/encode columns
    then split our data into 20/80, 
    then split the 80% into 30/70
    
    perform a train, validate, test split
    
    return: the three split pandas dataframes-train/validate/test
    """
      
    
    train_validate, test = train_test_split(df, test_size=0.2, random_state=3210, stratify=df.churn)
    train, validate = train_test_split(train_validate, train_size=0.7, random_state=3210, stratify=train_validate.churn)
    return train, validate, test

#create dummy columns:
def preprocess_telco(train, val, test):
    
    '''
    This function will create dummy columns for categorical varibales in the train, val, and test
    '''
    
    
    dummies_train_list = []
    dummies_val_list = []
    dummies_test_list = []
    cat_cols = ['gender',
                'senior_citizen',
                'partner',
                'dependents',
                'phone_service',
                'multiple_lines',
                'online_security',
                'online_backup',
                'device_protection',
                'tech_support',
                'streaming_tv',
                'streaming_movies',
                'paperless_billing',
                'contract_type',
                'internet_service_type',
                'payment_type']
    for col in cat_cols:
        dummies_train = pd.get_dummies(train[col], prefix=col, drop_first=True)
        dummies_train_list.append(dummies_train)
        dummies_val = pd.get_dummies(val[col], prefix=col, drop_first=True)
        dummies_val_list.append(dummies_val)
        dummies_test = pd.get_dummies(test[col], prefix=col, drop_first=True)
        dummies_test_list.append(dummies_test)
    
    train = pd.concat([train] + dummies_train_list, axis = 1)
   

    val = pd.concat([val]+ dummies_val_list, axis = 1)
    

    test = pd.concat([test]+ dummies_test_list, axis =1)
    


def prep_telco_data(df):
    """
    used to clean data for modeling
    """
    # Drop duplicate columns
    df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'], inplace=True)
       
    # Drop null values stored as whitespace    
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != '']
    
    # Convert to correct datatype
    df['total_charges'] = df.total_charges.astype(float)
    
    # Convert binary categorical variables to numeric
    df['gender_encoded'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partner_encoded'] = df.partner.map({'Yes': 1, 'No': 0})
    df['dependents_encoded'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['phone_service_encoded'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['paperless_billing_encoded'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['churn_encoded'] = df.churn.map({'Yes': 1, 'No': 0})
    
    # Get dummies for non-binary categorical variables
    dummy_df = pd.get_dummies(df[['multiple_lines', \
                              'online_security', \
                              'online_backup', \
                              'device_protection', \
                              'tech_support', \
                              'streaming_tv', \
                              'streaming_movies', \
                              'contract_type', \
                              'internet_service_type', \
                              'payment_type']], dummy_na=False, \
                              drop_first=True)
    
    # Concatenate dummy dataframe to original 
    df = pd.concat([df, dummy_df], axis=1)
    
    # split the data
    train, validate, test = train_validate_test()
    
    return train, validate, test
  
def drop_columns(train, val, test):
    train.set_index('customer_id', inplace=True)
    
    
    val.set_index('customer_id', inplace=True)
    
    
    test.set_index('customer_id', inplace=True)

    
    dropcols = ['gender',
                'senior_citizen',
                'partner',
                'dependents',
                'phone_service',
                'multiple_lines',
                'online_security',
                'online_backup',
                'device_protection',
                'tech_support',
                'streaming_tv',
                'streaming_movies',
                'paperless_billing',
                'contract_type',
                'internet_service_type',
                'payment_type',
               'churn']
    train.drop(columns = dropcols, inplace = True)
    val.drop(columns = dropcols, inplace=True)
    test.drop(columns = dropcols, inplace=True)
    return train, val, test

#---------------------------------------------------#
def get_metrics(mod, X, y, df):
    '''
    this functio will print out the metrics of a model
    '''
    baseline_accuracy = (df.did_churn == 0).mean()
    y_pred = mod.predict(X)
    accuracy = mod.score(X, y)
    prfs = pd.DataFrame(precision_recall_fscore_support(y, y_pred), index=['precision', 'recall', 'f1-score', 'support'])
    
    print(f'''
    BASELINE accuracy is: {baseline_accuracy:.2%}
    The accuracy for our model is: {accuracy:.2%} 
    ''')
    return prfs




def calculate_classification_metrics(TN, FP, FN, TP):
    '''
    will calculate TP, FP, FN, and TP for a model
    '''
    all_ = (TP + TN + FP + FN)

    accuracy = (TP + TN) / all_

    TPR = recall = TP / (TP + FN)
    FPR = FP / (FP + TN)

    TNR = TN / (FP + TN)
    FNR = FN / (FN + TP)

    precision = TP / (TP + FP)
    f1 = 2 * ((precision * recall) / (precision + recall))

    support_pos = TP + FN
    support_neg = FP + TN

    print(f"Accuracy: {accuracy}\n")
    print(f"True Positive Rate/Sensitivity/Recall/Power: {TPR}")
    print(f"False Positive Rate/False Alarm Ratio/Fall-out: {FPR}")
    print(f"True Negative Rate/Specificity/Selectivity: {TNR}")
    print(f"False Negative Rate/Miss Rate: {FNR}\n")
    print(f"Precision/PPV: {precision}")
    print(f"F1 Score: {f1}\n")
    print(f"Support (0): {support_pos}")
    print(f"Support (1): {support_neg}")







