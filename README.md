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

* We could observe that the residuals are pretty symmetrically distributed, and clustered around 0. In general, there are no clear patterns or trends, there exists no outliers,  which meams our model is good. 
* The (1,2) plot is the quantile plot for the residuals, that compares their distribution to that of a sample of independent normals. The residuals were really normal since this plot to be roughly on the diagonal.
* The (1,1) and (1,2) plots try address the constant variance assumptions. These plots don't have particular shapes which shows  variance is constant.
* Plot (2,2) detects outliers in the predictors, besides just looking at the actual values themselves, is through their leverage values. There is no particular outliers in our case. 
