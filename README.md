# Classification Project: Telco Churn Data Set:

### Project Description:
This project is to evaluate why customers are leaving Telco. Are they a bad company? Are the customers just not willing to pay anymore? This jupyter notebook will give a deep exploration of what could be driving churn in the customers through the steps in the data science pipeline.

### Project Goals: 
- Find drivers for customer churn at Telco. Why are they leaving Telco?
- Construct a ML classification model that accurately predicts customer churn.
- Present your process and findings to the lead data scientist.

### Initial Questions:
1. How often do people churn?
2. Is churn impacted by the type of service?
3. Is churn impacted by financial features?

### Data Dictionary
|**Feature**|**Definition**|
|----|----|
|`gender`| The gender of the customer. "Female" or "Male"|
|`senior citizen`| If the customer is a senior citizen or not. 0 = not senior, 1 = senior citizen  |
|`partner`| If the customer has a partner or not. No = single, yes = partner|
|`dependents`| If the customer is a dependent of an adult. No = not dependent, Yes = dependent|
|`tenure`| The time (in months) a customer has be with telco.|
|`phone service`| If the customer has phone service. Yes = phone service, No = no phone service |
|`multiple lines`| If the customer has multiple lines. Yes = do have multiple lines. No = Do not have multiple lines. No phone service = no phone service|
|`online security`| If the customer has online security. No = no online security, Yes = has online security, No internet service = no internet service|
|`tech_support`| If the customer has technical support. No = dont have tech support, Yes = have tech support, No internet service = no internet service|
|`streaming_tv`| If the customer has tv streaming. Yes = do have streaming, No = no streaming, no internet service = no internet service|
|`streaming_movies`| If the customer has movie streaming. yes = have streaming, no = no streaming|
|`paperless_billing`| If the customer has paperless billing. yes = use paperless billing. no = paperless billing|
|`monthly_charges`| The charges for the month.|
|`total_charges`| The total charges (month * tenure).|
|`churn`| If the customer has ended, churned, their service with Telco.|
|`contract_type`| The type of contract a customer has. |
|`internet_service_type`| The type of internet service a customer has. Fiber optic, DSL, None|
|`payment_type`| What type of payment method the customer use.|
|`did_chur`| This numerical column for churn. 0 = no_churn, 1 = churn|

### Project Planning: 
1. Data Acquisition
    - Acquire the data from the telco data set
2. Data Preparation
    - Clean the data: 
        - remove columns
        - encode dummies
        - create new features
        - fix datatypes
3. Exploratory data analysis and Hypothesis testing
    - explore the univariate, bivariate, and multivariate variables to create a hypothesis
    - test the hypothesis with proper statistical tests
4. Modeling:
    - do some preprocessing
    - find the best model for selected features out of:
        - Decision tree
        - Random forest
        - KNN
        - Logistic regression
    - Complete train, validate, test
5. Create Report


### How to use:
1. You can download the repository to your local device and run the all the files to understand the process.
    - I have created the notebook in a way where each step of the planning is laid out in a notebook to help with understanding:
        - A. Data Acquisition
        - B. Data Preparation
        - C. Exploratory analysis
        - D. Hypothesis testing
        - E. Modeling
2. If you would like to replicate the project without downloading this repository:
    - Make sure to look at the files to understand what each function does.
    - You will have to copy the wrangle functions to your own python file and import the functions.
    - You will need access to the code-up sql data library with your own credentials.
    
    
### Key findings: 
- This will be a list of key findings and takeaways: 

## Exploratory Analysis: Takeaways
1. Univariate:
    - Numerical: 
        - Total charges has a positive skew mostly around lower total charges.
            - There are a large amount of outliers
        - Monthly charges look to be more binomial, with peak around the lower charges. 
        - Tenure shows a higher amount of customers either being new or older but a lower amount in the mean tenure range. 
        - None of the data is normally distributed
    - Categorical: 
        - Customer_id can be removed from the data frame
        - Gender ratios are relatively the same
        - There is start difference between senior_citzen. 0 indicates that they are not a sernior_citizen while 1 indicates that they are. 
        - Partner counts are relatively the same
        - There are not a lot of dependents(individuals who are dependent on their parents) in the data. 
        - There is a large amount of observations that do have phone service compared to those that dont.
        - There is an even split between people who don’t have multiple phone lines compared to those who do, and a small portion of people that do not have phone service at all
        - online security has a higher amount of no's compared to those that do and those that do not have internet service
        - there is a more even distribution of device protection, but still has higher no's.
        - tech support also seems to be a feature that is not used as often.
        - streaming tv seems to comparatively the same
        - streaming movies seems to be comparatively the same 
        - more people do paperless billing
        - month to month contracts seem to be the most popular
        - fiber optic is the internet service that has the most customers
        - electronic check seems to be the most common payment method
2. Bivariate:
    - Numerical: 
        - Tenure does not seem to influence churn
        - **Monthly_Charges** is an important facotr to look at
        - Total charges also does not seem to influence churn
        - All the data, when a spearman test is run, have a monotonic relationship
    - Categorical: 
        - Gender:
            - no significance
        - **Senior Citizen**:
            - More elderly churn
            - significant difference
            - some relationship between churn
        - Partner:
            - more single people churn
            - some relationship between churn
        - Dependents:
            - Less dependents churn compared to dependents
            - some relationship between churn
        - Phone service:
            - no big difference
        - Multiple lines:
            - no big difference
        - Internet security
            - people with no security churn more often
        - Online backup
            - no online backup churned more often than yes, and no internet service
        - Device protection:
            - People who dont have device protection churn more
        - Tech support:
            - people with no tech support churn more often
        - Streaming Tv:
            - people who have no streaming for tv churn more often
        - Streaming movies:
            - people who have no movie streaming churn more often
        - Paperless billing
            - people who pay by paperless billing churn more
        - **contract type**:
            - significant churn rate for a month to month compared to others
        - **internet_sercive_type**:
            - people who use fiber churn more
        - **payment type**:
            - people who pay by electronic check chrun more
3. Hypothesis: 
    - Hypothesis 1: Are customers more likley to churn if they are a senior citizen <br>
        - $H_0$: Senior citizen and churn are independent of each other <br>
        - $H_a$: Senior citizen an churn are dependent on each other <br>
        - Answer: We reject the null hypothesis: Senior citizen and churn are independent of each other. Therefore, Senior citizen an churn are dependent on each other.
    - Hypothesis 2: Are customers more likley to churn due to contract type? <br>
        - $H_0$: Contract type and churn are independent of each other. <br>
        - $H_a$: Contract type an churn are dependent on each other. <br>
        - Answer: We reject the null hypothesis: Contract type and churn are independent of each other. Therefore, Contract type an churn are dependent on each other.
    - Hypothesis 3: Are customers more likley to churn due to payment type? <br>
        - $H_0$: Payment type and churn are independent of each other. <br>
        - $H_a$: Payment type an churn are dependent on each other. <br>
    - Hypothesis 4: Are customers more likley to churn due to payment type? <br>
        - $H_0$: Monthly charges and churn are have no relationship. <br>
        - $H_a$: Monthly charges and churn are have a monotonic relationship <br>
        - Answer: We reject H₀, there appears to be a monotonic relationship.
    - Hypothesis 5: Are customer more likley to churn due to payment type?
        - $H_0$: Internet type and churn are independent of each other. <br>
        - $H_a$: Internet type and churn are dependent on each other. <br>
        - Answer: We reject the null hypothesis: Internet type and churn are independent of each other. Therefore, Internet type and churn are dependent on each other.
        
### Recommendations:
Based on everything I looked at I would recommend:<br><br>
    - Figuring out a way to make services such as streaming more desirable. They could lead to even more profit.<br>
    - Looking at making services such as online protection and tech support more available or more advertising.<br> 
Based on churn:<br>
    - Talk to senior citizens and discuss methods that could make options to stay at telco easier. Maybe they are getting overwhelmed by technology or don't understand how to properly work with the services provided by Telco.<br>
    - Try lowering the cost of Fiber, it may be to expensive month to month which is why people may be churning. <br>
    - Advertise the other contracts with incentives or deals to make them more appeling to lower the amount of customers that use month-to-month service.<br>
