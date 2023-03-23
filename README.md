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
2. Features (# 1-7 above list of features) were set as variable y, whereas the reported loan_status was set as variable X
3. Further split data into training and testing data (i.e, X_train, X_test, y_train, and y_test)
4. Fit a logistics regression model with training data (i.e. X_train and y_train)
5. Evaluate model predictions via balanced accuracy and confusion matrix with testing data (i.e., X_test and y_test)
6. Re-sample the training data (i.e., X_resampled and y_resampled)
7. Re-fit the logistics regression model and evaluate predictions from re-sampled training data (#6 above) via balanced accuracy and confusion matrix
8. Compare the acuracy, precision, and recall of the two logistics regression models

## Results

## Logistics regression model results
Below are the results of evaluating the model predictions via balanced accuracy and confusion matrix (#5 in above workflow). Results are reported for both `0` healthy loans or `1` high-risk loans:
* *0 Precision* = 1.00
* *1 Precision* = 0.87
* *0 Recall* = 1.00
* *1 Recall* = 0.89
* *0 Accuracy (f1-score)* = 1.00
* *1 Accuracy (f1-score)* =  0.88

## Re-sampled and re-trained logistics regression model results
Below are the results of evaluating the model predictions after resampling via balanced accuracy and confusion matrix (#7 in above workflow). Results are reported for both `0` healthy loans or `1` high-risk loans::
* *0 Precision* = 1.00
* *1 Precision* = 0.87
* *0 Recall* = 1.00
* *1 Recall* = 1.00
* *0 Accuracy (f1-score)* = 1.00
* *1 Accuracy (f1-score)* =  0.93

## Summary
The logistic regression model did a great job at predicting low-risk loans off the bat. The precision and accuracy did not change for low-risk loans (`0`) as it was perfect at predicting for both models both models were perfect (1.0). However, for high-risk loans (`1`), the model with re-sampled/oversamples the data did a much better job at predicting high-risk loans. For this model, the precision remained the same at 0.87 among both models, meaning that they were consistent in making false positive predictions of high-risk loans. However, the recall (aka accuracy) increased from 0.89 to 1.0. This means it got better at NOT making false negative predictions. As a result the harmonic mean of precision and accuracy (i.e. the f1-score) also increased from 0.88 to 0.93.

These data suggests that there is still a "risk" in its ability to predit high-risk loans. However, because of the fact that A) the re-sampled logidtics regression model improved the ability for prediction and that B) the "inaccuracy" is a conservative measure of prediction (i.e., meaning that it may slightly over-predict high risk loans), I would absolutley recommend the use and implentation of this re-sampled logistics regression model for use.


# thanks - Nioclette :)
