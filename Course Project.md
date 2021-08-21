
Course Project: Practical Machine Learning
DBaca
date: "8/18/2021

 
### Project Description

#### Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement, a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.  

---

#### Project Goal

In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbbell of 6 participants. The 6 participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The goal of this project is to predict the classification manner in which they performed their exercises. This is the "classe" variable (A, B, C, D, E) in the training set. 

The outcome variable, "classe", is a factor variable with 5 levels. In this data set, participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in 5 different ways:

* Class A: exactly according to specification    
* Class B: throwing the elbows to the front    
* Class C: lifting the dumbbell only halfway    
* Class D: lowering the dumbbell only halfway    
* Class E: throwing the hips to the front    

This project report will describe:  
1) how the machine learning models were built   
2) how cross validation was used   
3) calculation of the expected Out-of-Sample-Error (OOSE)            
4) why the choice of machine learning model was made     
5) application of machine learning algorithm to the 20 test cases available in the test data and submission of predictions in appropriate format to the Course Project Prediction Quiz for automated grading

---
 
#### Model Discussion    

Boosting, random forests, and ensembling models have proven to be some of the best prediction modeling tools that achieve success in prediction challenges. Random forests and Boosting are usually the top two performing algorithms. Random forest models can be difficult to interpret but are often very accurate. Random Forest and Gradient Boosting Machine, along with the CART Decision Tree model, will be explored in this project.

---

#### Data Source

The **training** data for this project is available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv     
The **test** data is available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv     
Additional information for this project is available here:      http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset)

#### Data Loading

library(caret)
library(ggplot2)
library(rattle)
library(randomForest)

set.seed(213)
train_csv <- read.csv("data/pml-training.csv")
test_csv <- read.csv("data/pml-testing.csv")
dim(test_csv)

--- 

#### Exploratory Analysis and Data Cleaning
Prior to downloading the data in R, the pml-training.csv and pml-testing.csv files was opened and examined, and predictor columns containing empty, limited, irrelevant, and near zero variance data were identified. 


#### Removing first 7 columns of data that contain irrelevant predictor information
train_csv <- train_csv[ , -c(1:7)]
dim(train_csv)

#### Removing data columns containing any missing variables (or NA)
train_csv <- train_csv[ , colSums(is.na(train_csv)) == 0] 
dim(train_csv)

#### diagnosis of near zero variance: vector of integers corresponding to the column positions with problematic predictors
nzv <- nearZeroVar(train_csv, saveMetrics = FALSE)
nzv

#### Removing near zero variance problematic predictors, essentially all remaining column names starting with "kurtosis", "skewness", "min", "max", and "amplitude"
train_csv <- train_csv[ , -nzv]

#### Converting training$classe into a factor variable

train_csv$classe <- as.factor(train_csv$classe)
str(train_csv)
table(train_csv$classe)


--- 

#### Creating training and testing data sets

inTrain <- createDataPartition(y=train_csv$classe, p=0.7, list=FALSE)
training <- train_csv[inTrain, ]
testing <- train_csv[-inTrain, ]

#### Examining dimensions of training and testing data sets
dim(training)
dim(testing)


--- 

### Creating and Testing Machine Learning Models

Models used for this project are: 

* CART Decision Tree     
* Gradient Boosted Machine     
* Random Forest      

#### CART (Classification and Regression Trees) Decision Tree model   
Decision Tree algorithms can be used for classification or regression (numeric) predictive modeling problems. In classification models, such as this one, CART uses Gini Impurity calculations in the process of splitting the dataset into a simplified decision tree model.

We will apply the k-fold Cross Validation method to test the effectiveness on all our prediction models. k-fold Cross Validation generally results in low bias and low variance in the  model because it involves splitting the dataset into k-subsets and averaging observations over the training data a number of times to thus derive a more accurate prediction of model performance.
 
 
#### k-fold CV trainControl() parameters for model fit, k=10
#### random search for mtry, the number of variables randomly sampled as available for splitting at each tree node
cart_Control <- trainControl(method = "cv", number = 10, search = "random")

cart_model <- train(classe ~ ., data=training, method = "rpart", preProcess=c("center", "scale"), 
                    trControl=cart_Control)
cart_model


--- 

#### CART Plot

fancyRpartPlot(cart_model$finalModel, sub="Classification and Regression Tree")


--- 

#### CART Prediction on **testing** data

cart_predict <- predict(cart_model, newdata=testing)


--- 

#### CART Confusion Matrix on **testing** data

cart_confmatrix <- confusionMatrix(cart_predict, testing$classe)
cart_confmatrix


--- 

#### CART Accuracy and Out-of-Sample-Error on **testing** data

cart_Accu <- cart_confmatrix$overall[[1]]

cart_Accu

cart_OOSE <- 1 - cart_confmatrix$overall[[1]]

cart_OOSE


---

#### Gradient Boosting Machine (GBM)
In gradient boosting, the  machine learning procedure is to consecutively fit an ensemble of less accurate models in iterations thereby building on successive improving approximations to produce a more accurate estimate of the response variable. 

#### k-fold CV trainControl() parameters for model fit, k=10
gbm_Control <- trainControl(method = "cv", number = 10)

gbm_model <- train(as.factor(classe) ~ ., data=training, method = "gbm",
                         preProcess=c("center", "scale"), trControl = gbm_Control, verbose=FALSE)

gbm_model


---

#### GBM Plot

plot(gbm_model)


Comparison of Accuracy based on Boosting iterations and Max Tree Depth

---

#### GBM Prediction on **testing** data

pred_gbm <- predict(gbm_model, newdata=testing)


--- 

#### GBM Confusion Matrix on **testing** data

gbm_confmatrix <- confusionMatrix(pred_gbm, testing$classe)
gbm_confmatrix


--- 

#### GBM Accuracy and Out-of-Sample-Error on **testing** data

gbm_Accu <- gbm_confmatrix$overall[1] 

gbm_Accu

gbm_OOSE <- 1 - gbm_confmatrix$overall[[1]] 

gbm_OOSE


---

#### Random Forest Model   
Random forest classification algorithms build an ensemble of multiple decision trees and averages them together to get a more accurate and stable prediction while decreasing the variance of the model.

#### k-fold CV trainControl() parameters for model fit, k=10
rf_Control <- trainControl(method = "cv", number = 10)

rf_model <- randomForest(classe ~ ., data=training, trControl=rf_Control, 
                         preProcess=c("center", "scale"))

rf_model



---

#### Random Forest Plot

plot(rf_model, main="Random Forest Error vs Number of Trees")


---

#### Random Forest variable Importance Plot

importance(rf_model)
order(importance(rf_model), decreasing=TRUE)
colnames(training[order(importance(rf_model), decreasing=TRUE)])

varImpPlot(rf_model, main="Random Forest Variable Importance")


For the Random Forest classification model, the node impurity is measured by the Gini index. MeanDecreaseGini coefficients are a measure of the variable contribution to the homogeneity of the nodes and leaves in the resulting random forest model. The higher MeanDecreaseGini score, the higher the variable importance to the model, in this case, "roll_belt" has the highest MeanDecreaseGini score (861).

---

#### Random Forest prediction with **testing** data

pred_rf <- predict(rf_model, newdata=testing)


---

#### Random Forest confusion matrix

rf_confmatrix <- confusionMatrix(pred_rf, testing$classe)
rf_confmatrix


---

#### Random Forest Accuracy and Out-of-Sample-Error on **testing** data

rf_Accu <- rf_confmatrix$overall[1]

rf_Accu

rf_OOSE <- 1 - rf_confmatrix$overall[[1]]

rf_OOSE


---

### Model Evaluation 
Among these 3 models, Random Forest has the best performance Accuracy ( > 99.5%) closely followed by Gradient Boosting Machine (Accuracy > 96.3%). 
Accuracy among the testing set samples, where the two methods agree, is calculated for Random Forest model and Gradient Boosting Machine.

pred_rf <- predict(rf_model, newdata=testing)

pred_gbm <- predict(gbm_model, newdata=testing)

predDF <- data.frame(pred_rf, pred_gbm, y = testing$classe)

#### Accuracy among the testing set samples where the two methods agree

sum(pred_rf[predDF$pred_rf == predDF$pred_gbm] == predDF$y[predDF$pred_rf == 
                        predDF$pred_gbm])  /  sum(predDF$pred_rf == predDF$pred_gbm)

There is accuracy agreement of 99.7% between GBM and Random Forest on the testing data set.


---

#### Evaluation of model accuracies and Out-of-Sample-Errors (OOSE) on **testing** data
The Out of Sample Error is error rate on a testing data set,          
the In Sample Error is the error rate on the training data set.         
In Sample Error is always less than Out Of Sample error, the reason being overfitting.

Model_Accuracies <- data.frame(Model = c('CART', 'GBM', 'RF'), 
                               Accuracy = rbind(cart_Accu[1], gbm_Accu[1], rf_Accu[1]), 
                               OOSE = rbind(cart_OOSE[1], gbm_OOSE[1], rf_OOSE[1]))
                               
Model_Accuracies

Comparing the 3 models tested, the CART model has the lowest accuracy (87.7%), the Random Forest model has the highest accuracy (99.56%), and thus lowest Out-of-Sample-Error (OOSE)(0.441%).

---

### Prediction
Prediction based on Random Forest model on 20 cases of **test_csv** data

predict(rf_model, newdata=test_csv)


### Conclusion

The goal of this project was to build and compare machine learning models to predict the classification manner in which the participants performed their exercises using data from accelerometers on the belt, forearm, arm, and dumbbell. I have explored and reported on three models in this project, Random Forest, Gradient Boosting Machine, and the CART Decision Tree model with good results.

There is an inherent bias-variance trade-off in the modeling process and k-fold Cross Validation parameters are an effective way to reduce both bias and variance in building and testing the models. k-fold Cross Validation methods were performed to calculate each model's Accuracy and Out-of-Sample-Error on the first-seen testing data, then collected and assembled for comparison.

My choice of machine learning model was Random Forest based on the highest Accuracy (99.56%) and lowest Out-of-Sample-Error achieved on the held-out and unknown testing data of the three models tested. In view of the exceptional high Accuracy of the Random Forest model, further training improvements, such as combined ensembling models, have been deemed unnecessary in this instance. 

---