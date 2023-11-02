# CREDIT RISK ANALYSIS 
For this activity, I used lending data to build a machine-learning model that evaluates borrowers and identifies their creditworthiness.

## An Overview of The Analysis: 

1) I used the lending_data.csv, which contains loan_size, interest_rate, borrower_income, debt_to_income, num_of_accounts, derogatory_marks, total_debt, and loan_status. The loan_status column contains either 0 or 1, where 0 means that the loan is healthy, and 1 means that the loan is at a high risk of defaulting. I stored the data in a dataframe.

2) To estimate creditworthiness, first, I stored the labels set from the loan_status column in the y variable. Then, I stored the features DataFrame (all the columns except loan_status) in the X variable. I checked the balance of the labels with value_counts. The results showed that, in our dataset, 75036 loans were healthy and 2500 were high-risk.

3) Then, I used the train_test_split module from sklearn to split the data into training and testing variables, these are: X_train, X_test, y_train, and y_test. And I assigned a random_state of 1 to the function to ensure that the train/test split is consistent, the same data points are assigned to the training and testing sets across multiple runs of code.

4) Then, I created a Logistic Regression Model with the original data. I used LogisticRegression(), from sklearn, with a random_state of 1. I fit the model with the training data, X_train and y_train, and predicted on testing data labels with predict() using the testing feature data, X_test, and the fitted model, lr_model.

5) Following this, I calculated the accuracy score of the model with balanced_accuracy_score() from sklearn, I used y_test a d testing_prediction to obtain the accuracy.

6) Next, I generated a confusion matrix for the model with confusion_matrix() from sklearn, based on y_test and testing_prediction.

7) Next, I obtained a classification report for the model with classification_report() from sklearn, and I used y_test and testing_prediction.

8) Then, I used RandomOverSampler() from imbalanced-learn to resample the data. I fit the model with the training data, X_train and y_train. I generated resampled data, X_resampled and y_resampled, and used unique() to obtain the count of distinct values in the resampled labels data.

9) Next, I created a Logistic Regression Model with the resampled data, fit the data, and made predictions. 

10) Lastly, I obtained the accuracy score, confusion matrix, and classification report of the resampled model.



## The Results: 

1. Machine Learning Model 1: (Original Data) 
    * The Accuracy for Model 1 was **0.952**
    * The Precision for Healthy Loans is **1.00** and Precision for high-risk loans is **0.85**
    * The Recall for Healthy Loans is **0.99** and Recall for high-risk loans is **0.91**

2. Machine Learning Model 2: (Over-sampled Data) 
    * The Accuracy for Model 2 was **0.995**
    * The Precision for Healthy Loans is **0.99** and Precision for high-risk loans is **0.99**
    * The Recall for Healthy Loans is **0.99** and Recall for high-risk loans is **0.99**

## Summary:

Overall, it would appear that the model with the over-sampled Data yielded better accuracy for both loans. However, to recommend a model, we would need to know the question we are trying to answer. 

### What is the likelihood of a Loan being good? 
To answer this question, the Logistic Regression model based on the original Data set can be used, because it has a higher Precision(1.00) than the model based on the over-sampled Data (0.99). With the recall values being the same (0.99) for both, we can conclude that the original dataset would yield better insights when trying to identify loans that will be good. 

### What is the likelihood of a loan being defaulted? 
To answer this question, the Logistic Regression model based on the over-sampled Data set can be used, because it has a higher Precision(0.99) and Recall(0.99) than the Precision (0.85) and Recall(0.91) values based on the original data. Therefore, we can conclude that the original dataset would yield better insights when trying to identify the likelihood of default. 

## Conclusion: Personally, I would think knowing the likelihood of Default would be more important for lenders, so I would recommend the Model 2, based on the over-sampled Data. 
