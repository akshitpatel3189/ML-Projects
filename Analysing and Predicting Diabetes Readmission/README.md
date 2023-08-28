#  Analysing and Predicting Diabetes Readmission

This is a group project that is part of my master's program. In this project, we analyze diabetes patients' data and predict how many times these patients have to be readmitted to the hospital again.

# Tools and Libraries

I am using Jupyter Notebook for this project.
For running this project I have used the following Python libraries
- pandas
- ploty
- NumPy
- matplotlib
- SMOTE
- torch


# Description:

I have used the Diabetes 130-US hospitals for the years 1999-2008 dataset which is provided in the dataset_link.txt file.

- First I perform **exploratory data analysis**. In this, I have grouped some columns with  additional sunset datasets. I have removed some columns which have more than 60% of null values. Then I check the accuracy of the preprocessed dataset by SMOTE, SVC, and Classification Report.
- After preprocessing I run and compare **four models** to see which model is best for prediction. Models are Random Forest Classifier, XGBoost Classifier, Linear Regression, and Gradient Boosting.
- I **tuned hyperparameters** by ensemble modeling techniques Bagging and Boosting Classifier and compared results of 4 models with F1, precision, recall, and accuracy score.
