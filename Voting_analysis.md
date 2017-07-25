# Voting analysis
Mahalakshmi  



Loading the relevent Libraries


```r
library(dplyr)
library(tidyr)
library(rpart)
library(randomForest)
library(arm)
library(ramify)
library(MLmetrics)
library(ROCR)
```

### LOADING THE DATA 


```r
train <- read.csv("traindata_R.csv")
```

### Part-1 - DATA MANIPULATION AND SUMMARY STATISTICS
Grouping the data based on education and the summary data  with 20 values  is stored in a single data frame -train_grp_edu 


```r
train_grp_edu <- train %>% group_by(Wife_education)%>%
                summarise(
                Count=n(),
                Avg_age = mean(Wife_age),
                Avg_Child = round(mean(Number_of_children_ever_born)),
                Percent_Working =round(sum(Wife_working)*100/n(),2),
                Percent_HSL = round(sum(Standard_of_living_index==4)*100/n(),2)
                )
#printing  the 20 values in Long format from a single DF
print (train_grp_edu %>% gather("Variables","Values",2:6))
```

```
## # A tibble: 20 × 3
##    Wife_education       Variables    Values
##             <int>           <chr>     <dbl>
## 1               1           Count 104.00000
## 2               2           Count 219.00000
## 3               3           Count 279.00000
## 4               4           Count 380.00000
## 5               1         Avg_age  38.09615
## 6               2         Avg_age  31.45662
## 7               3         Avg_age  30.22222
## 8               4         Avg_age  33.42632
## 9               1       Avg_Child   4.00000
## 10              2       Avg_Child   3.00000
## 11              3       Avg_Child   3.00000
## 12              4       Avg_Child   3.00000
## 13              1 Percent_Working  75.96000
## 14              2 Percent_Working  76.26000
## 15              3 Percent_Working  78.49000
## 16              4 Percent_Working  69.47000
## 17              1     Percent_HSL  23.08000
## 18              2     Percent_HSL  32.88000
## 19              3     Percent_HSL  33.69000
## 20              4     Percent_HSL  67.11000
```

### Part-2 -MODELLING AND PREDICTING


```r
#converting the Response variable into a Factor variable
train$Party_voted_for <- as.factor(train$Party_voted_for)

#Decision Tree using R
DT_model <- rpart(Party_voted_for~.,data=train,method = "class")
#Random Forest using R
RF_model <- randomForest(Party_voted_for~. ,data=train)
#Logistic regression using R
LR_model <- glm(Party_voted_for~.,data=train,family = binomial(link = "logit"))
```

Loading the test Dataset and predicting the values


```r
#Predictions
test<- read.csv("testdata_R.csv")
x_test <- test[,-10]
y_test <- test$Party_voted_for
#Predictions 

DT_probs <- predict(DT_model,x_test)
#The index with highest probability is chosen and mapped with voting choice
DT_output <- argmax(DT_probs)-1
RF_output <- predict(RF_model,x_test)
Lr_prob <- invlogit(predict(LR_model,x_test))
LR_output <- 1*(Lr_prob>0.5)
```

#### Confusion Matrix and Accuracy rates


```
## [1] "Confusion Matrix for Decision Trees:"
```

```
##       y_pred
## y_true   0   1
##      0 102 105
##      1  43 241
```

```
## [1] "Accuracy of Decision Trees: 0.698574"
```



```
## [1] "Confusion Matrix for Random Forest:"
```

```
##       y_pred
## y_true   0   1
##      0 104 103
##      1  44 240
```

```
## [1] "Accuracy of Random Forest: 0.700611"
```



```
## [1] "Confusion Matrix for Logistic Regression:"
```

```
##       y_pred
## y_true   0   1
##      0 101 106
##      1  36 248
```

```
## [1] "Accuracy of Logistic Regression: 0.710794"
```

### Part-3 MODEL EVALUATION


```r
DT_output.pr <- predict(DT_model,x_test,type="prob")[,2]
ROC_DT <- performance(prediction(DT_output.pr,y_test),"tpr","fpr")
RF_output.pr <- predict(RF_model,x_test,type="prob")[,2]
ROC_RF <- performance(prediction(RF_output.pr,y_test),"tpr","fpr")
ROC_LR <- performance(prediction(Lr_prob,y_test),"tpr","fpr")
#Plotting the ROC curves
plot(ROC_DT,col="blue",main="ROC CURVES FOR VARIOUS MODELS")
abline(a=0,b=1,lty=3, col="grey")
plot(ROC_RF,col="red",add=TRUE)
plot(ROC_LR,col="green",add=TRUE)
```

![](Voting_analysis_files/figure-html/ROC_PLOTTING-1.png)<!-- -->


#### OBSERVATION:
From the AUC values the Random Forest model turns out to be better than the decision tree and Logistic regression.


```r
auc_DT <-performance(prediction(DT_output.pr,y_test),measure = "auc")
print(auc_DT@y.values)
```

```
## [[1]]
## [1] 0.6959924
```

```r
auc_RF <-performance(prediction(RF_output.pr,y_test),measure = "auc")
print(auc_RF@y.values)
```

```
## [[1]]
## [1] 0.743332
```

```r
auc_LR <-performance(prediction(Lr_prob,y_test),measure = "auc")
print(auc_LR@y.values)
```

```
## [[1]]
## [1] 0.7176124
```

