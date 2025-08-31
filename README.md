* Python 3.9.7
* scikit-learn 1.0.2
* xgboost 1.5.1
* numpy 1.22.0
* nltk 3.6.7
* pandas 1.3.5
* Flask 2.0.2
* Bootstrap CDN 5.1.3


* Dataset and Attribute description are available under dataset\ folder
* Data Cleaning, Visualization and Text Preprocessing (NLP) are applied on the dataset. TF-IDF Vectorizer is used to vectorize the textual data(review_title+review_text). It measures the relative importance of the word w.r.to other documents
* Dataset suffers from Class Imbalance Issue and SMOTE Oversampling technique is used before applying the model
* Machine Learning Classification Models (Logistic Regression, Naive Baiyes, Tree Algorithms : (Decision Tree, Random Forrest, xgboost) are applied on the vectorized data and the target column (user_sentiment). the objective of this ML model is to classify the sentiment to positive(1) or negative(0). Best Model is selected based on the various ML classification metrics (Accuracy, Precision, Recall, F1 Score, AUC). xgboost is selected to be a better model based on the evaluation metrics.
*  Colloborative Filtering Recommender system is created based on User-user and item-item approaches.RMSE evaluation metric is used for the evaluation.
*  \SentimentBasedProductRecommendation.ipynb Jupiter notebook contains the code for Sentiment Classification and Recommender Systems
*  Top 20 products are filtered using the better recommender system, and for each of the products predicted the user_sentiment for all the reviews and filtered out the Top 5 products that have higher Postive User Sentiment (model.py)
*  Machine Learning models are saved in the pickle files(under the pickle\); Flask API (app.py) is used to interface and test the Machine Learning models. Bootstrap and Flask jinja templates (templates\index.html) are used for setting up the User interface. No additional Custom Styles used.


How to run this application:

install all the required packages from the requirements.txt file: 

from the base folder: go to :- cd SentimentBasedProductRecommendation-main

1) go to app.py file: 

2) on terminal run: python .\app.py 

3) this will give a localhost : run on your browser to use it: 



#  if any conflict occurs ensure your ML model and flask model have same version of packages:





=======
# Sentiment-Analysis
>>>>>>> cde9816ca1c7f7d65c293c5577d21c06f48d22df
