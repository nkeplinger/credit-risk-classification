# Credit-Risk-Classification with Supervised Machine Learning

## ANalysis Overview:

# Training and Testing Dataset: 
For this challenge I used the class-provided input dataset of historical lending activity from a peer-to-peer lending services company found here: `Resources/lending_data.csv`. This dataset  includes data over 77k reported lenders used to build a supervised machine learning model that can identify the creditworthiness of borrowers . Each of these >77k datasets are classified by loan status as either `0` healthy loans or `1` high-risk loans. There are seven features from each lender that were used for training and predicting the model. These include: 
1. Loan size
2. Interest rate
3. Borrower Income
4. Debt-to-Income ratio
5. Number of Accounts
6. Derogatory Marks
7. Total debt

# Logistic Regression Model 
Below is the workflow overview for creating a logistics regression model for testing and evaluating this dataset (code found here: `credit_risk_classification.ipynb`):
1. Data was split into Training and Testing 
2. My features (i.e. 1-7 above) were set as variable y, whereas the reported loan_status was set as variable X
3. Further split data into training and testing data (i.e, X_train, X_test, y_train, and y_test)
4. Fit a logistics regression model with training data (i.e. X_train and y_train)
5. Evaluate model via balanced accuracy and confusion matrix with testing data (i.e., X_test and y_test)



