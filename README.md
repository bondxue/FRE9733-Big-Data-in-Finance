# FRE9733-Big-Data-in-Finance
homeworks for big data course

**Update soon...**

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
## homework 2:  customer churn prediction using logistic regression
Build a logistic regression model to predict `churn` using `Mocked_Customer_Data_With_Missing.xlsx` dataset.
#### Dataset 
`Mocked_Customer_Data_With_Missing.xlsx`
#### Summary of model 
### Comparsion Using different data transforming
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
## homework 3: 

