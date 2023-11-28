
# Credit card fraud detection using Ensemble methods

In this project, I am detecting fraud transactions of credit cards using ensemble methods.


# Tools and Libraries

I am using Jupyter Notebook for this project.
For running this project I have used the following Python libraries
-   pandas
-   NumPy
-   plotly
-   seaborn
-   matplotlib
-   sklearn


# Description:

I have used the Credit Card Fraud dataset which is provided in the dataset_link.txt file.
I have preprocessed a dataset by eliminating some columns by using Heatmap correlation. Then I have done feature importance by Random Forest classifier for viewing which columns are best with the target column.

Now from the preprocessed dataset, I use the Random Forest classifier model to achieve more accuracy. Then I tuned hyperparameters by using the Bagging and Bossing Classifier and compared results with F1, Precision, and Recall scores. then with the confusion matrix I compared predicted labels and original labels and downloaded with new CSV file.
