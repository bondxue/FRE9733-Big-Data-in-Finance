# Homework 1 - longley dataset testing
# Mengheng Xue
# Date: 02/06/2019 

# step 1: load "longley" into new data frame called "test"
test = data.frame(longley)

# step 2: find dimensions and variable names
str(longley) # check the structure of the data
summary(longley) # summarize the dataset
dim(test) # 16 7
nrow(test) # 16
ncol(test) # 7
colnames(test) # "GNP.deflator" "GNP" "Unemployed" "Armed.Forces" "Population" "Year" "Employed"

# step 3: create histogram fro Armed.Forces
armed_forces = test[, 4]
hist(armed_forces)

# step 4: create a correlation matrix of all variables
correlation_matrix = cor(test)
correlation_matrix 

# step 5: conduct multiple linear regression, "Employed" as dependent variable, the rest as independent variables
# We consider test dataset as our Training set 
training_set = test

# Fitting Multiple Linear Regression to the Training set
regressor = lm(formula = Employed ~., data = training_set)
summary(regressor)

# Building the optimal model using Backward Elimination 
# select the significance level to stay in the model (SL = 0.05)
# p value of GNP.deflator is highest (0.863141), remove this predictor 
regressor = lm(formula = Employed ~ GNP + Unemployed + Armed.Forces + Population + Year,
               data = training_set)
summary(regressor)

# p value of Population is 0.641607, remove it 
regressor = lm(formula = Employed ~ GNP + Unemployed + Armed.Forces + Year,
               data = training_set)
summary(regressor)

# Predicting the test set results
employed_pred = predict(regressor, newdata = test)

# Summarize accuracy base on MSE 
MSE = mean((test$Employed - employed_pred)^2)
print(MSE)


# step 6: intepret the result
# Summary of our model:
# 1. All p values of predictors are smaller than 0.05, we stop. Here is our final model.
# 2. Our final predictors are "Unemployed", "Armed.Forces", "Year" and "GNP".
# 3. Since we have high R-square and low p-value, this means our model explains a lot of variation in the data and is also significant. 
# 4. We also find that P values of other predictors will change when we remove a certain predictor.  

# step 7: visualize the results
# ====== plot of residual analysis and outlier detection ========
# we could observe that the residuals are pretty symmetrically distributed, and clustered around 0 
# in general there are no clear patterns or trends, there exists no outliers,  which meams our model is good. 

# The (1,2) plot is the quantile plot for the residuals, that compares their distribution to that of a sample of independent normals.
# The residuals were really normal since this plot to be roughly on the diagonal.

# The (1,1) and (1,2)  try address the constant variance assumptions. 
# These plots don't have particular shapes which shows  variance is constant.

# Plot(2,2) detect outliers in the predictors,
# besides just looking at the actual values themselves, is through their leverage values. 
# There is no particular outliers in our case. 
par(mfrow=c(2,2)) # set up ploting environment (2,2)
plot(regressor, pch=23, bg='orange',cex=2) 

