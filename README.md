Hello, Team, these is a sentiment analysis on a given review, a web application build on prodcut reviews called Amazon Echo. I downloaded the product reviews from kaggle and performed data cleaning like text cleaning 
applied various analysis like the distributions of data i mean since it is a text data and the data contains other coulmns such as rating and feedbacks, tried check the distribution of 
sentiment of the review according to the review and rating.

- handled null values
- cleaed data using stemming to better understanding by the model

As the problem statement is of binary classification the feedback 0 says negative and feedbacj 1 says positive, hence used Random Forest and XGBoost algorithms useing GrdiSearch for better 
paramters so that the model can learn better

Used Radom Forest model as final model, the results are better compared to XGBoost, pickeled the file deployed using streamlit and render services.

