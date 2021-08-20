Course Project: Practical Machine Learning
DBaca
8/18/2021
Project Description
Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement, a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

Project Goal
In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbbell of 6 participants. The 6 participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The goal of this project is to predict the classification manner in which they performed their exercises. This is the “classe” variable (A, B, C, D, E) in the training set.

The outcome variable, “classe”, is a factor variable with 5 levels. In this data set, participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in 5 different ways:

Class A: exactly according to specification
Class B: throwing the elbows to the front
Class C: lifting the dumbbell only halfway
Class D: lowering the dumbbell only halfway
Class E: throwing the hips to the front
This project report will describe:
1) how the machine learning models were built
2) how cross validation was used
3) calculation of the expected Out-of-Sample-Error (OOSE)
4) why the choice of machine learning model was made
5) application of machine learning algorithm to the 20 test cases available in the test data and submission of predictions in appropriate format to the Course Project Prediction Quiz for automated grading

Model Discussion
Boosting, random forests, and ensembling models have proven to be some of the best prediction modeling tools that achieve success in prediction challenges. Random forests and Boosting are usually the top two performing algorithms. Random forest models can be difficult to interpret but are often very accurate. Random Forest and Gradient Boosting Machine, along with the CART Decision Tree model, will be explored in this project.

Data Source
The training data for this project is available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
The test data is available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
Additional information for this project is available here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset)

Data Loading
library(caret)
library(ggplot2)
library(rattle)
library(randomForest)

set.seed(213)
train_csv <- read.csv("data/pml-training.csv")
test_csv <- read.csv("data/pml-testing.csv")
dim(test_csv)
## [1]  20 160
Exploratory Analysis and Data Cleaning
Prior to downloading the data in R, the pml-training.csv and pml-testing.csv files was opened and examined, and predictor columns containing empty, limited, irrelevant, and near zero variance data were identified.

# Removing first 7 columns of data that contain irrelevant predictor information
train_csv <- train_csv[ , -c(1:7)]
dim(train_csv)
## [1] 19622   153
# Removing data columns containing any missing variables (or NA)
train_csv <- train_csv[ , colSums(is.na(train_csv)) == 0] 
dim(train_csv)
## [1] 19622    86
# diagnosis of near zero variance: vector of integers corresponding to the column positions with problematic predictors
nzv <- nearZeroVar(train_csv, saveMetrics = FALSE)
nzv
##  [1]  5  6  7  8  9 10 11 12 13 36 37 38 39 40 41 45 46 47 48 49 50 51 52 53 67
## [26] 68 69 70 71 72 73 74 75
# Removing near zero variance problematic predictors, essentially all remaining column names starting with "kurtosis", "skewness", "min", "max", and "amplitude"
train_csv <- train_csv[ , -nzv]

# Converting training$classe into a factor variable
train_csv$classe <- as.factor(train_csv$classe)
str(train_csv)
## 'data.frame':    19622 obs. of  53 variables:
##  $ roll_belt           : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt          : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ gyros_belt_x        : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
##  $ gyros_belt_y        : num  0 0 0 0 0.02 0 0 0 0 0 ...
##  $ gyros_belt_z        : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
##  $ accel_belt_x        : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
##  $ accel_belt_y        : int  4 4 5 3 2 4 3 4 2 4 ...
##  $ accel_belt_z        : int  22 22 23 21 24 21 21 21 24 22 ...
##  $ magnet_belt_x       : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
##  $ magnet_belt_y       : int  599 608 600 604 600 603 599 603 602 609 ...
##  $ magnet_belt_z       : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
##  $ roll_arm            : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
##  $ pitch_arm           : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
##  $ yaw_arm             : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm     : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ gyros_arm_x         : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
##  $ gyros_arm_y         : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
##  $ gyros_arm_z         : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
##  $ accel_arm_x         : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
##  $ accel_arm_y         : int  109 110 110 111 111 111 111 111 109 110 ...
##  $ accel_arm_z         : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
##  $ magnet_arm_x        : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
##  $ magnet_arm_y        : int  337 337 344 344 337 342 336 338 341 334 ...
##  $ magnet_arm_z        : int  516 513 513 512 506 513 509 510 518 516 ...
##  $ roll_dumbbell       : num  13.1 13.1 12.9 13.4 13.4 ...
##  $ pitch_dumbbell      : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
##  $ yaw_dumbbell        : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
##  $ total_accel_dumbbell: int  37 37 37 37 37 37 37 37 37 37 ...
##  $ gyros_dumbbell_x    : num  0 0 0 0 0 0 0 0 0 0 ...
##  $ gyros_dumbbell_y    : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 ...
##  $ gyros_dumbbell_z    : num  0 0 0 -0.02 0 0 0 0 0 0 ...
##  $ accel_dumbbell_x    : int  -234 -233 -232 -232 -233 -234 -232 -234 -232 -235 ...
##  $ accel_dumbbell_y    : int  47 47 46 48 48 48 47 46 47 48 ...
##  $ accel_dumbbell_z    : int  -271 -269 -270 -269 -270 -269 -270 -272 -269 -270 ...
##  $ magnet_dumbbell_x   : int  -559 -555 -561 -552 -554 -558 -551 -555 -549 -558 ...
##  $ magnet_dumbbell_y   : int  293 296 298 303 292 294 295 300 292 291 ...
##  $ magnet_dumbbell_z   : num  -65 -64 -63 -60 -68 -66 -70 -74 -65 -69 ...
##  $ roll_forearm        : num  28.4 28.3 28.3 28.1 28 27.9 27.9 27.8 27.7 27.7 ...
##  $ pitch_forearm       : num  -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.8 -63.8 -63.8 ...
##  $ yaw_forearm         : num  -153 -153 -152 -152 -152 -152 -152 -152 -152 -152 ...
##  $ total_accel_forearm : int  36 36 36 36 36 36 36 36 36 36 ...
##  $ gyros_forearm_x     : num  0.03 0.02 0.03 0.02 0.02 0.02 0.02 0.02 0.03 0.02 ...
##  $ gyros_forearm_y     : num  0 0 -0.02 -0.02 0 -0.02 0 -0.02 0 0 ...
##  $ gyros_forearm_z     : num  -0.02 -0.02 0 0 -0.02 -0.03 -0.02 0 -0.02 -0.02 ...
##  $ accel_forearm_x     : int  192 192 196 189 189 193 195 193 193 190 ...
##  $ accel_forearm_y     : int  203 203 204 206 206 203 205 205 204 205 ...
##  $ accel_forearm_z     : int  -215 -216 -213 -214 -214 -215 -215 -213 -214 -215 ...
##  $ magnet_forearm_x    : int  -17 -18 -18 -16 -17 -9 -18 -9 -16 -22 ...
##  $ magnet_forearm_y    : num  654 661 658 658 655 660 659 660 653 656 ...
##  $ magnet_forearm_z    : num  476 473 469 469 473 478 470 474 476 473 ...
##  $ classe              : Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
table(train_csv$classe)
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
Creating training and testing data sets
inTrain <- createDataPartition(y=train_csv$classe, p=0.7, list=FALSE)
training <- train_csv[inTrain, ]
testing <- train_csv[-inTrain, ]

# Examining dimensions of training and testing data sets
dim(training)
## [1] 13737    53
dim(testing)
## [1] 5885   53
Creating and Testing Machine Learning Models
Models used for this project are:

CART Decision Tree
Gradient Boosted Machine
Random Forest
CART (Classification and Regression Trees) Decision Tree model
Decision Tree algorithms can be used for classification or regression (numeric) predictive modeling problems. In classification models, such as this one, CART uses Gini Impurity calculations in the process of splitting the dataset into a simplified decision tree model.

We will apply the k-fold Cross Validation method to test the effectiveness on all our prediction models. k-fold Cross Validation generally results in low bias and low variance in the model because it involves splitting the dataset into k-subsets and averaging observations over the training data a number of times to thus derive a more accurate prediction of model performance.

# k-fold CV trainControl() parameters for model fit, k=10
# random search for mtry, the number of variables randomly sampled as available for splitting at each tree node
cart_Control <- trainControl(method = "cv", number = 10, search = "random")

cart_model <- train(classe ~ ., data=training, method = "rpart", preProcess=c("center", "scale"), 
                    trControl=cart_Control)
cart_model
## CART 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered (52), scaled (52) 
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 12363, 12362, 12362, 12363, 12365, 12365, ... 
## Resampling results across tuning parameters:
## 
##   cp           Accuracy   Kappa     
##   0.001830943  0.8711520  0.83692342
##   0.004068762  0.8131329  0.76344861
##   0.114637372  0.3229260  0.05883736
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was cp = 0.001830943.
CART Plot
fancyRpartPlot(cart_model$finalModel, sub="Classification and Regression Tree")


CART Prediction on testing data
cart_predict <- predict(cart_model, newdata=testing)
CART Confusion Matrix on testing data
cart_confmatrix <- confusionMatrix(cart_predict, testing$classe)
cart_confmatrix
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1601   72   13   20   10
##          B   25  916   68   31   39
##          C   24   70  891   42   44
##          D   10   32   43  828   61
##          E   14   49   11   43  928
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8775          
##                  95% CI : (0.8688, 0.8858)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8449          
##                                           
##  Mcnemar's Test P-Value : 4.546e-08       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9564   0.8042   0.8684   0.8589   0.8577
## Specificity            0.9727   0.9657   0.9630   0.9703   0.9756
## Pos Pred Value         0.9330   0.8489   0.8319   0.8501   0.8880
## Neg Pred Value         0.9825   0.9536   0.9720   0.9723   0.9682
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2720   0.1556   0.1514   0.1407   0.1577
## Detection Prevalence   0.2916   0.1833   0.1820   0.1655   0.1776
## Balanced Accuracy      0.9645   0.8849   0.9157   0.9146   0.9167
CART Accuracy and Out-of-Sample-Error on testing data
cart_Accu <- cart_confmatrix$overall[[1]]  
cart_Accu
## [1] 0.8774851
cart_OOSE <- 1 - cart_confmatrix$overall[[1]]
cart_OOSE
## [1] 0.1225149
Gradient Boosting Machine (GBM)
In gradient boosting, the machine learning procedure is to consecutively fit an ensemble of less accurate models in iterations thereby building on successive improving approximations to produce a more accurate estimate of the response variable.

# k-fold CV trainControl() parameters for model fit, k=10
gbm_Control <- trainControl(method = "cv", number = 10)

gbm_model <- train(as.factor(classe) ~ ., data=training, method = "gbm",
                         preProcess=c("center", "scale"), trControl = gbm_Control, verbose=FALSE)

gbm_model
## Stochastic Gradient Boosting 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered (52), scaled (52) 
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 12363, 12363, 12363, 12364, 12364, 12363, ... 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.7510362  0.6841212
##   1                  100      0.8215766  0.7741445
##   1                  150      0.8536801  0.8148260
##   2                   50      0.8530237  0.8137928
##   2                  100      0.9056545  0.8805862
##   2                  150      0.9308424  0.9124685
##   3                   50      0.8932078  0.8647765
##   3                  100      0.9409609  0.9252949
##   3                  150      0.9616360  0.9514648
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were n.trees = 150, interaction.depth =
##  3, shrinkage = 0.1 and n.minobsinnode = 10.
GBM Plot
plot(gbm_model)


Comparison of Accuracy based on Boosting iterations and Max Tree Depth

GBM Prediction on testing data
pred_gbm <- predict(gbm_model, newdata=testing)
GBM Confusion Matrix on testing data
gbm_confmatrix <- confusionMatrix(pred_gbm, testing$classe)
gbm_confmatrix
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1645   36    0    0    1
##          B   22 1070   31    3    9
##          C    4   31  982   26   15
##          D    2    1   13  930   15
##          E    1    1    0    5 1042
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9633         
##                  95% CI : (0.9582, 0.968)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9536         
##                                          
##  Mcnemar's Test P-Value : 1.078e-05      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9827   0.9394   0.9571   0.9647   0.9630
## Specificity            0.9912   0.9863   0.9844   0.9937   0.9985
## Pos Pred Value         0.9780   0.9427   0.9282   0.9677   0.9933
## Neg Pred Value         0.9931   0.9855   0.9909   0.9931   0.9917
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2795   0.1818   0.1669   0.1580   0.1771
## Detection Prevalence   0.2858   0.1929   0.1798   0.1633   0.1782
## Balanced Accuracy      0.9869   0.9629   0.9707   0.9792   0.9808
GBM Accuracy and Out-of-Sample-Error on testing data
gbm_Accu <- gbm_confmatrix$overall[1] 
gbm_Accu
##  Accuracy 
## 0.9632965
gbm_OOSE <- 1 - gbm_confmatrix$overall[[1]]  
gbm_OOSE
## [1] 0.03670348
Random Forest Model
Random forest classification algorithms build an ensemble of multiple decision trees and averages them together to get a more accurate and stable prediction while decreasing the variance of the model.

# k-fold CV trainControl() parameters for model fit, k=10
rf_Control <- trainControl(method = "cv", number = 10)

rf_model <- randomForest(classe ~ ., data=training, trControl=rf_Control, 
                         preProcess=c("center", "scale"))

rf_model
## 
## Call:
##  randomForest(formula = classe ~ ., data = training, trControl = rf_Control,      preProcess = c("center", "scale")) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.52%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3904    2    0    0    0 0.0005120328
## B   12 2640    6    0    0 0.0067720090
## C    0   18 2375    3    0 0.0087646077
## D    0    0   19 2231    2 0.0093250444
## E    0    0    1    9 2515 0.0039603960
Random Forest Plot
plot(rf_model, main="Random Forest Error vs Number of Trees")


Random Forest variable Importance Plot
importance(rf_model)
##                      MeanDecreaseGini
## roll_belt                   861.09471
## pitch_belt                  460.35683
## yaw_belt                    617.86197
## total_accel_belt            166.85127
## gyros_belt_x                 67.98574
## gyros_belt_y                 80.44523
## gyros_belt_z                210.55341
## accel_belt_x                 82.16769
## accel_belt_y                 90.74322
## accel_belt_z                280.35787
## magnet_belt_x               172.62251
## magnet_belt_y               261.39806
## magnet_belt_z               284.00812
## roll_arm                    223.27187
## pitch_arm                   125.32384
## yaw_arm                     188.61764
## total_accel_arm              69.79201
## gyros_arm_x                  91.59082
## gyros_arm_y                  93.95655
## gyros_arm_z                  42.40090
## accel_arm_x                 161.61730
## accel_arm_y                 107.64574
## accel_arm_z                  93.10391
## magnet_arm_x                188.80284
## magnet_arm_y                164.38154
## magnet_arm_z                130.21533
## roll_dumbbell               309.20508
## pitch_dumbbell              118.74279
## yaw_dumbbell                184.67219
## total_accel_dumbbell        184.07091
## gyros_dumbbell_x             89.09560
## gyros_dumbbell_y            180.80547
## gyros_dumbbell_z             55.59114
## accel_dumbbell_x            172.53093
## accel_dumbbell_y            286.81456
## accel_dumbbell_z            238.12819
## magnet_dumbbell_x           327.78257
## magnet_dumbbell_y           466.94744
## magnet_dumbbell_z           531.88596
## roll_forearm                428.22513
## pitch_forearm               540.82529
## yaw_forearm                 125.19164
## total_accel_forearm          81.21232
## gyros_forearm_x              57.54404
## gyros_forearm_y              87.24037
## gyros_forearm_z              63.43451
## accel_forearm_x             226.15878
## accel_forearm_y             101.08685
## accel_forearm_z             172.43095
## magnet_forearm_x            155.59267
## magnet_forearm_y            152.46971
## magnet_forearm_z            205.07278
order(importance(rf_model), decreasing=TRUE)
##  [1]  1  3 41 39 38  2 40 37 27 35 13 10 12 36 47 14  7 52 24 16 29 30 32 11 34
## [26] 49  4 25 21 50 51 26 15 42 28 22 48 19 23 18  9 31 45  8 43  6 17  5 46 44
## [51] 33 20
colnames(training[order(importance(rf_model), decreasing=TRUE)])
##  [1] "roll_belt"            "yaw_belt"             "pitch_forearm"       
##  [4] "magnet_dumbbell_z"    "magnet_dumbbell_y"    "pitch_belt"          
##  [7] "roll_forearm"         "magnet_dumbbell_x"    "roll_dumbbell"       
## [10] "accel_dumbbell_y"     "magnet_belt_z"        "accel_belt_z"        
## [13] "magnet_belt_y"        "accel_dumbbell_z"     "accel_forearm_x"     
## [16] "roll_arm"             "gyros_belt_z"         "magnet_forearm_z"    
## [19] "magnet_arm_x"         "yaw_arm"              "yaw_dumbbell"        
## [22] "total_accel_dumbbell" "gyros_dumbbell_y"     "magnet_belt_x"       
## [25] "accel_dumbbell_x"     "accel_forearm_z"      "total_accel_belt"    
## [28] "magnet_arm_y"         "accel_arm_x"          "magnet_forearm_x"    
## [31] "magnet_forearm_y"     "magnet_arm_z"         "pitch_arm"           
## [34] "yaw_forearm"          "pitch_dumbbell"       "accel_arm_y"         
## [37] "accel_forearm_y"      "gyros_arm_y"          "accel_arm_z"         
## [40] "gyros_arm_x"          "accel_belt_y"         "gyros_dumbbell_x"    
## [43] "gyros_forearm_y"      "accel_belt_x"         "total_accel_forearm" 
## [46] "gyros_belt_y"         "total_accel_arm"      "gyros_belt_x"        
## [49] "gyros_forearm_z"      "gyros_forearm_x"      "gyros_dumbbell_z"    
## [52] "gyros_arm_z"
varImpPlot(rf_model, main="Random Forest Variable Importance")


For the Random Forest classification model, the node impurity is measured by the Gini index. MeanDecreaseGini coefficients are a measure of the variable contribution to the homogeneity of the nodes and leaves in the resulting random forest model. The higher MeanDecreaseGini score, the higher the variable importance to the model, in this case, “roll_belt” has the highest MeanDecreaseGini score (861).

Random Forest prediction with testing data
pred_rf <- predict(rf_model, newdata=testing)
Random Forest confusion matrix
rf_confmatrix <- confusionMatrix(pred_rf, testing$classe)
rf_confmatrix
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    6    0    0    0
##          B    0 1131    3    0    0
##          C    0    2 1020    8    0
##          D    0    0    3  955    3
##          E    0    0    0    1 1079
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9956          
##                  95% CI : (0.9935, 0.9971)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9944          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9930   0.9942   0.9907   0.9972
## Specificity            0.9986   0.9994   0.9979   0.9988   0.9998
## Pos Pred Value         0.9964   0.9974   0.9903   0.9938   0.9991
## Neg Pred Value         1.0000   0.9983   0.9988   0.9982   0.9994
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1922   0.1733   0.1623   0.1833
## Detection Prevalence   0.2855   0.1927   0.1750   0.1633   0.1835
## Balanced Accuracy      0.9993   0.9962   0.9960   0.9947   0.9985
Random Forest Accuracy and Out-of-Sample-Error on testing data
rf_Accu <- rf_confmatrix$overall[1] 
rf_Accu
## Accuracy 
## 0.995582
rf_OOSE <- 1 - rf_confmatrix$overall[[1]]
rf_OOSE
## [1] 0.004418012
Model Evaluation
Among these 3 models, Random Forest has the best performance Accuracy ( > 99.5%) closely followed by Gradient Boosting Machine (Accuracy > 96.3%). Accuracy among the testing set samples, where the two methods agree, is calculated for Random Forest model and Gradient Boosting Machine.

pred_rf <- predict(rf_model, newdata=testing)
pred_gbm <- predict(gbm_model, newdata=testing)

predDF <- data.frame(pred_rf, pred_gbm, y = testing$classe)
# Accuracy among the testing set samples where the two methods agree
sum(pred_rf[predDF$pred_rf == predDF$pred_gbm] == predDF$y[predDF$pred_rf == 
                        predDF$pred_gbm])  /  sum(predDF$pred_rf == predDF$pred_gbm)
## [1] 0.9968315
There is accuracy agreement of 99.7% between GBM and Random Forest on the testing data set.

Evaluation of model accuracies and Out-of-Sample-Errors (OOSE) on testing data
The Out of Sample Error is error rate on a testing data set,
the In Sample Error is the error rate on the training data set.
In Sample Error is always less than Out Of Sample error, the reason being overfitting.

Model_Accuracies <- data.frame(Model = c('CART', 'GBM', 'RF'), 
                               Accuracy = rbind(cart_Accu[1], gbm_Accu[1], rf_Accu[1]), 
                               OOSE = rbind(cart_OOSE[1], gbm_OOSE[1], rf_OOSE[1]))
Model_Accuracies
##   Model  Accuracy        OOSE
## 1  CART 0.8774851 0.122514868
## 2   GBM 0.9632965 0.036703483
## 3    RF 0.9955820 0.004418012
Comparing the 3 models tested, the CART model has the lowest accuracy (87.7%), the Random Forest model has the highest accuracy (99.56%), and thus lowest Out-of-Sample-Error (OOSE)(0.441%).

Prediction
Prediction based on Random Forest model on 20 cases of test_csv data

predict(rf_model, newdata=test_csv)
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
Conclusion
The goal of this project was to build and compare machine learning models to predict the classification manner in which the participants performed their exercises using data from accelerometers on the belt, forearm, arm, and dumbbell. I have explored and reported on three models in this project, Random Forest, Gradient Boosting Machine, and the CART Decision Tree model with good results.

There is an inherent bias-variance trade-off in the modeling process and k-fold Cross Validation parameters are an effective way to reduce both bias and variance in building and testing the models. k-fold Cross Validation methods were performed to calculate each model’s Accuracy and Out-of-Sample-Error on the first-seen testing data, then collected and assembled for comparison.

My choice of machine learning model was Random Forest based on the highest Accuracy (99.56%) and lowest Out-of-Sample-Error achieved on the held-out and unknown testing data of the three models tested. In view of the exceptional high Accuracy of the Random Forest model, further training improvements, such as combined ensembling models, have been deemed unnecessary in this instance. The predictions based on the Random Forest model on 20 cases of test_csv data were 100% accurate.

