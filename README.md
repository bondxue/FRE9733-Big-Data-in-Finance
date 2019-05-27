# FRE9733-Big-Data-in-Finance

-----------------------
## homework 1: longley dateset analysis using linear regression
In R, load the `longley` dataset, conduct *linear regression* by using `Employed` as dependent variable and the rest as independent variables. 
#### Dataset 
The `Longley` dataset contains various US macroeconomic variables that are known to be highly collinear. It has been used to appraise the accuracy of least squares routines.

#### Summary of model 
* Using backward elimination, our final predictors are `Unemployed`, `Armed.Forces`, `Year` and `GNP`. All *p values* of predictors are smaller than 0.05.
* Since we have high *R-square* and low *p-value*, this means our model explains a lot of variation in the data and is also significant. 
* We also find that *p-values* of other predictors will change when we remove a certain predictor. 

#### Plot of residual analysis and outlier detection
<img src="https://github.com/bondxue/FRE9733-Big-Data-in-Finance/blob/master/homework1-longleyDatasetTest/images/Rplot.png" width="500">

* We could observe that the residuals are pretty symmetrically distributed, and clustered around 0. In general, there are no clear patterns or trends, there exists no outliers,  which meams our model is good. 
* The (1,2) plot is the quantile plot for the residuals, that compares their distribution to that of a sample of independent normals. The residuals were really normal since this plot to be roughly on the diagonal.
* The (1,1) and (1,2) plots try address the constant variance assumptions. These plots don't have particular shapes which shows  variance is constant.
* Plot (2,2) detects outliers in the predictors, besides just looking at the actual values themselves, is through their leverage values. There is no particular outliers in our case. 

-----------------------
## homework 3:  customer churn prediction using logistic regression
Build a logistic regression model to predict `churn` using `Mocked_Customer_Data_With_Missing.xlsx` dataset.
#### Dataset 
`Mocked_Customer_Data_With_Missing.xlsx`
#### Model 
Logistic regression 
#### Comparsion Using different data transforming
We try to use different *data transformation* techniques to improve prediction performance. The outcomes are displayed in the following table. We can see that: 
+ Generally, our *logistic* model performs very well, obtaining both *precision* and *recall* larger than 0.9 on the test set.  
+ We use *normalization* then *PCA* to improve the performance. However, it  does not change so much.  We think the raw data features are already well structured features and obtain excellent results, there is limited room to improve the performance by data transformation. 
+ We only discuss the performance of our model in terms of prediction accuracy here. We think data transformation is more important in terms of time cost of training, which will make gradient decent faster than we only use the raw data. Here, however, since our data size is relatively small and well structured (relatively close, no extreme outliers), it may be not obvious that the model is faster when we apply data transformation. 

| data transform | Precision | Recall | F1 score |
| --- | --- | --- | --- |
| raw data | 0.932 | 0.917 | 0.925 |
| normalize | 0.934 | 0.922 | 0.928 |
| PCA | 0.936 | 0.921 | 0.928 |

-----------------------
## homework 5: loan fallout prediction 
#### Dataset
`Loan_FallOut_Prediction.csv`

#### Models
We use five classification models, i.e., **Naïve Bayesian**, **Decision Tree**, **SVM**, **Logistic Regression** and **ANN** to predict loan fallout. 

#### Compare the model performance by ROC, Precision, Recall, and F-Score
<img src="https://github.com/bondxue/FRE9733-Big-Data-in-Finance/blob/master/homework5-LoanFalloutPrediction/images/roc.PNG" width="1000">


| | LR | NB |  SVM  | ANN | DT |
| --- | --- | --- | --- | --- | --- |
| precision | 0.589 | 0.649 | 0.660 | 0.942 | 0.773 |
| recall | 0.437 | 0.243 | 0.701 | 0.728 | 0.783 |
| F1 | 0.502 | 0.354 | 0.680 | 0.821 | 0.778 |
| cv score mean | 0.606 | 0.603 | 0.692 | 0.854 | 0.796 |
| cv score std | 0.002 | 0.004 | 0.003 | 0.004 | 0.002 |

#### Summary of results 
+ **ANN** has much greater prediction ability than the other models. If we keep tuning the parameters, we believe the performance can be improved even better.
+ The second best is the **Decision Tree**. It is simple, but it has unexpected powerful prediction ability (AUC = 0.80).
+ The third is the **SVM**. We use a *linear kernel* first, but its performance is really bad. Then we adopt *rbf kernel* and achieve good performance due to the problem is nonlinear.
+ We would say that **LR** and **NB** performances are really similar based on ROC curves and cross-validation scores. We could see from the cross-validation score. (LR = 0.606 and NB = 0.603)
+ Also, we find that feature engineering is really important for classification problems. Good feature selection and missing data handling may improve the model performance extremely.

-------------------------------------
## homework 7: customer clusteriong 
Conduct **K-Means** and **Hierarchical Clustering** algorithms to segment customers in `Mocked_Customer_Data.xlsx`.

#### Visualizing the k-mean clusters
<img src="https://github.com/bondxue/FRE9733-Big-Data-in-Finance/blob/master/homework7-Part1-CustomerClustering/images/k-mean.PNG" width="700">

* For visulizing purpose, we only choose `normalized yahoo air` and `normalized total air` to draw the pair-wise plot to show our *k-mean clustering* performance.
* One improvement method is that we could use **PCA** to reduce the dimension of input features， which can reduce the correlation between each features then achieve better clustering performance.

#### Dendrogram plot for hierarchical clustering 
<img src="https://github.com/bondxue/FRE9733-Big-Data-in-Finance/blob/master/homework7-Part1-CustomerClustering/images/dendrogram.PNG" width="700">

+ Since it takes so long to plot *dendrogram* for the whole dataset, I just use the first 1000 data to demonstrate dendrogram.
+ We could see that if we just use the first 10000 data, 2 clusters may be the best choices.


-------------------------------------------
## homework 11: sentiment analysis model and topic model for moive reviews 
### part 1: sentiment analysis model
#### Dataset
Download 2000 reviews from the following link:

http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz

#### Model 
Build a Sentiment Analysis model by comparing three classifiers, i.e., **Logistic regression**, **Naive Bayes**, and **SVM**. 

#### Word importance visualization 
For **LR**：

<img src="https://github.com/bondxue/FRE9733-Big-Data-in-Finance/blob/master/homework11-SentimentAnalysisMovieReview/images/LR.PNG" width="800">

For **SVM**:

<img src="https://github.com/bondxue/FRE9733-Big-Data-in-Finance/blob/master/homework11-SentimentAnalysisMovieReview/images/SVM.PNG" width="800">

#### Summary 
Based on our experiments, we find that **logistic regression** achieves the best accuracy performance at 0.86, which is very promising since we only choose the simple parameter setting and haven't done *regularzation*. We can continue to improve the model accuracy by regularization. Also, our dataset only contains 2000 reviews. We believe we can improve our model performance if we can extend our dataset.

### part 2: topic modeling 
I adopt three topic models to group 2000 reviews
1. **Nonnegative Matrix Factorization (NMF) with term frequency (tf)**, the topics are  
```Python
  Topic #0: family / love story
  Topic #1: negative feedbacks about time and characters
  Topic #2: positive feedbacks about characters
  Topic #3: star wars
  Topic #4: positive feedbacks about directors
  Topic #5: positive / negative feedback 
  Topic #6: television series
  Topic #7: The Black Cauldron
  Topic #8: jackie chan
  Topic #9: Alien
 ```
2. **Latent Dirichlet Allocation (LDA) with tf**

   **LDA with tf** is not well performed, all 2000 reviews are clustered into 
  ```Python
  Topic #6： positive review
  ```
3. **Nonnegative Matrix Factorization (NMF) with term frequency-inverse document frequency (tf_idf)**, the topics are:
```Python
  Topic #0: positive feedback
  Topic #1: Alien
  Topic #2: positive feedbacks about characters 
  Topic #3: Jackie Chan
  Topic #4: Scream
  Topic #5: The Truman Show
  Topic #6: Star Wars
  Topic #7: Clayton 
  Topic #8: Vampires
  Topic #9: Super
 ```
 
 #### Summary
+ **NMF with tf** performaces ok with some unclear topics.
+ **LDA with tf** is not doing so good in this study case, all reviews are grouped into one topic and the topic is not clear. 
+ **NMF with tf-idf** is the best model with no obsure optics and even correctly associated a topic with a specific movie.

----------------------------------------------
## final project： social media sentiment analysis

### Abstract
The goal of this project was to predict sentiment for 3000+ tweets by calling **Twitter's API** using Python. Sentiment analysis can predict many different emotions attached to the text, but in this report only 3 major were considered: `positive`, `neutral`, and `negative`. The training dataset was small (just over 3000 examples) and the data within it was highly skewed, which greatly impacted on the difficulty of building good classifier. I  utilize **bag-of-words** and apply both **Navie Bayes** and **Random Forest** classifictation, the classification accuracy at level of 88.5% was achieved.

### Extract data from Twitter API
Download 3000+ Tweets by calling Twitter's API and save them into a corpus CSV file.
<img src="https://github.com/bondxue/FRE9733-Big-Data-in-Finance/blob/master/FinalProject-SocialMediaSentimentAnalysis/images/download_tweets.PNG" width="700">


### Data preprocessing: data cleaning 
For the purpose of cleansing, the TwitterCleanup class was created. It consists methods allowing to execute all of the tasks show in the list above. Most of those is done using regular expressions. The class exposes it's interface through iterate() method - it yields every cleanup method in proper order.

+ Remove URLs
+ Remove usernames (mentions)
+ Remove tweets with *Not Available* text
+ Remove special characters
+ Remove numbers
+ Remove emojis
+ Remove News alerts like: `WATCH`, `LIVE` etc. 

<img src="https://github.com/bondxue/FRE9733-Big-Data-in-Finance/blob/master/FinalProject-SocialMediaSentimentAnalysis/images/modified_tweets.PNG" width="700">

### Data preprocessing: data lebeling
After cleaning text, using TextBlob API to tag each tweet into Positive, Negative, or Neutral. 
<img src="https://github.com/bondxue/FRE9733-Big-Data-in-Finance/blob/master/FinalProject-SocialMediaSentimentAnalysis/images/distribution.PNG" width="700">


### Create dataset with positive & negative sentiment for classification 
+ positive sentiment: 1
+ negative sentiment: 0
<img src="https://github.com/bondxue/FRE9733-Big-Data-in-Finance/blob/master/FinalProject-SocialMediaSentimentAnalysis/images/dataset_class.PNG" width="500">

### Create bag of words model
 For the text processing, **nltk** library is used. Stemming is done using **PorterStemmer** as the tweets are 100% in english.
 
### Classification Model Performance 
#### Navie Bayes

<img src="https://github.com/bondxue/FRE9733-Big-Data-in-Finance/blob/master/FinalProject-SocialMediaSentimentAnalysis/images/NavieBayes.PNG" width="500">

Result with accuracy at level of 76.7% seems to be quite nice result for such basic algorithm like Naive Bayes (having in mind that random classifier would yield result of around 50% accuracy). This performance may not hold for the final testing set. In order to see how the NaiveBayes performs in more general cases, 10-fold crossvalidation is used. This result still looks optimistic.

#### Random Forest 
<img src="https://github.com/bondxue/FRE9733-Big-Data-in-Finance/blob/master/FinalProject-SocialMediaSentimentAnalysis/images/randomForest.PNG" width="500">

We can see that the random forest performs even better than Navie Bayes with accuracy 88.5%.

#### Plot of ROC curve 

<img src="https://github.com/bondxue/FRE9733-Big-Data-in-Finance/blob/master/FinalProject-SocialMediaSentimentAnalysis/images/ROC.PNG" width="700">

We can see that **Random Forest** is indeed better than **Navie Bayes** in our case in terms of *AUC*.



#### Summary 
+ Experiment showed that prediction of text sentiment is a non-trivial task for machine learning. A lot of preprocessing is required just to be able to run any algorithm and see - usually not great - results.
+ Main problem for sentiment analysis is to craft the machine representation of the text. Simple **bag-of-words** was definitely not enough to obtain satisfying results, thus a lot of additional features were created basing on common sense (`number of emoticons`, `exclamation marks` etc.). Also, **Word2vec**  representation can raise the predictions quality.

+ Since it contained *highly skewed data* (small number of negative cases), the improvement in classfication accuracy for the given training dataset may be probably in the order of a few percent. 

+ The other thing that could possibly improve classification results will be to add a lot of additional examples (increase training dataset), because given 3000+ examples obviously do not contain every combination of words usage, moreover - a lot of emotion-expressing words surely are missing.



