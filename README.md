# liver-disease-prediction

Machine Learning project.
This repo is correlated to my other repo "data-visualization" where the data analysis and visualization part is handled.
Identify patients with liver disease based on several biochemical indicators, age and gender.
The dataset can be found on kaggle:
https://www.kaggle.com/datasets/uciml/indian-liver-patient-records/code?datasetId=2607&sortBy=voteCount

For this project i chose to concentrate on Logistic Regression, a simple decision tree model and a Support Vector Machine model.
The goal of this is to apply stratified K-fold cross validation to find the best performing model, and then apply again 
K-fold cross validation for hyperparameters tuning and asses if the chosen model is good for the given task, based on 
accuracy, precision, recall, F1-score.
