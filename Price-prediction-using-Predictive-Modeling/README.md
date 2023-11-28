# Price prediction using Predictive Modeling

In this project, I am predicting the price of US AirBnb using predictive modeling techniques.


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

I have used the US Airbnb open dataset which is provided in the dataset_link.txt file.
I have preprocessed a dataset by eliminating null values and filling some null values with appropriate column values. Furthermore, I make feature reports for categorical and continuous values. Then I did feature visualization by plotting some graphs. After that, I selected some columns by using Heatmap correlation.

Now from the preprocessed dataset, I compare different predictive models such as andom Forest, XGBoost, Linear Regrassion, Gradient Boosting, and MLP  Regrassion with MSE, RMSE, and R-squared (R2) Score to find which model is best. Then I tuned hyper perameter with GridSearchCV for better accuracy. Finally I plot the graph with the original and predictive price with the best model.
