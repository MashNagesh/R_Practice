#R - Case study for Emplay

#Relevant Libraries
library(dplyr)
library(tidyr)
library(rpart)
library(randomForest)
library(arm)
library(ramify)
library(MLmetrics)
library(ROCR)

#reading the training data
train <- read.csv("traindata_R.csv")

#Part-1 - DATA MANIPULATION AND SUMMARY STATISTICS
#grouping based on education
train_grp_edu <- train %>% group_by(Wife_education)%>%
                summarise(
                  Count=n(),
                  Avg_age = mean(Wife_age),
                  Avg_Child = round(mean(Number_of_children_ever_born)),
                  Percent_Working =round(sum(Wife_working)*100/n(),2),
                  Percent_HSL = round(sum(Standard_of_living_index==4)*100/n(),2)
                  )

#Converting the output to a single frame of 20 values
train_grp_edu <- train_grp_edu %>% gather("Variables","Values",2:6)

#Part-2 -MODELLING AND PREDICTING

#converting the Response variable into a Factor variable
train$Party_voted_for <- as.factor(train$Party_voted_for)

#Decision Tree using R
DT_model <- rpart(Party_voted_for~.,data=train,method = "class")
#Random Forest using R
RF_model <- randomForest(Party_voted_for~. ,data=train)
#Logistic regression using R
LR_model <- glm(Party_voted_for~.,data=train,family = binomial(link = "logit"))

#Predictions
test<- read.csv("testdata_R.csv")
x_test <- test[,-10]
y_test <- test$Party_voted_for

DT_probs <- predict(DT_model,x_test)
#The index with highest probability is chosen and mapped with voting choice
DT_output <- argmax(DT_probs)-1
RF_output <- predict(RF_model,x_test)
Lr_prob <- invlogit(predict(LR_model,x_test))
LR_output <- 1*(Lr_prob>0.5)

#Creating Confusion Matrix 
print ("Confusion Matrix for Decision Trees:")
ConfusionMatrix(DT_output,y_test)
sprintf ("Accuracy of Decision Trees: %f",Accuracy(DT_output,y_test))

print ("Confusion Matrix for Random Forest:")
ConfusionMatrix(RF_output,y_test)
sprintf ("Accuracy of Random Forest: %f",Accuracy(RF_output,y_test))

print ("Confusion Matrix for Logistic Regression:")
ConfusionMatrix(LR_output,y_test)
sprintf ("Accuracy of Logistic Regression: %f",Accuracy(LR_output,y_test))

#Part-3 Model Evaluation
#ROC curves
DT_output.pr <- predict(DT_model,x_test,type="prob")[,2]
ROC_DT <- performance(prediction(DT_output.pr,y_test),"tpr","fpr")
RF_output.pr <- predict(RF_model,x_test,type="prob")[,2]
ROC_RF <- performance(prediction(RF_output.pr,y_test),"tpr","fpr")
ROC_LR <- performance(prediction(Lr_prob,y_test),"tpr","fpr")

#Plotting the ROC curves
plot(ROC_DT,col="blue")
abline(a=0,b=1,lty=3, col="grey")
plot(ROC_RF,col="red",add=TRUE)
plot(ROC_LR,col="green",add=TRUE)

#getting the auc values
auc_DT <-performance(prediction(DT_output.pr,y_test),measure = "auc")
print(auc_DT@y.values)
auc_RF <-performance(prediction(RF_output.pr,y_test),measure = "auc")
print(auc_RF@y.values)
auc_LR <-performance(prediction(Lr_prob,y_test),measure = "auc")
print(auc_LR@y.values)

#tuning the RF_model
bestmtry <- tuneRF(train[,-10], train[,10], stepFactor=1.5, improve=1e-5, ntree=500)
print(bestmtry)
RF_model1 <- randomForest(Party_voted_for~. ,data=train,mtry=3)
RF_output1 <- predict(RF_model1,x_test)
Accuracy(RF_output1,y_test)
