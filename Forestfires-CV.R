library(e1071)
library(caret)
library(caTools)
library(kernlab)

data1 = read.csv('E:/BDAP/Datasets/forestfires.csv')
summary(data1)
nrow(data1)
str(data1)
plot(data1$area)
data1$area = log(data1$area+1)
plot(data1$area)

ind = createDataPartition(data1$area,p=2/3,list = FALSE)

trainset <- data1[ind,]
testset <- data1[-ind,]

controlparameter <- trainControl(method = 'cv',number = 10,savePredictions = TRUE,classProbs = TRUE)
parametergrid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.5, 1, 1.5, 2,5))
  
modelSVM <- train(area~temp+RH+wind+rain,data=trainset,method = 'svmLinear',trControl =controlparameter,tuneGrid =parametergrid)
modelSVM


predicts <- predict(modelSVM,testset)
RMSE(predicts,testset$area)

