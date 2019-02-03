library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)
library(rattle)
library(mlbench)
library(rpart)
library(gbm)
library(parallel)
library(doParallel)

set.seed(12345)

trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))


inTrain <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
myTraining <- training[inTrain, ]; myTesting <- training[-inTrain, ]
dim(myTraining); dim(myTesting)

cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster)

### can not deal with 160 variables and first we are going to cleanup the data
### then do a feature
#### exploratory Analysis to identify the key variables
table(myTraining$classe) 
prop.table(table(myTraining$classe)) 
prop.table(table(myTraining$user_name)) 
prop.table(table(myTraining$user_name,myTraining$classe),2) 


#### treat missing values in object
nzvTraining <- nearZeroVar(myTraining, saveMetrics=TRUE)
trainingSub <- myTraining[,(nzvTraining$nzv==FALSE)&(nzvTraining$zeroVar==FALSE)]

### now the above is 130 variables
## below is getting the columns out of the df which is not a feature
trainingSub <- trainingSub[,7:length(colnames(trainingSub))]

# Drop columns that have more than 20% NAs
miss <- colSums(is.na(trainingSub))
miss_prec <- miss/nrow(trainingSub)*100
col_miss <- names(miss_prec[miss_prec>20])

trainingSub <- trainingSub[,!(names(trainingSub) %in% col_miss)]
keepCols <- colnames(trainingSub[, -53]) #remove classe as it's not contained in testing
testingSub <- myTesting[keepCols] #keep only variables in testing
dim(trainingSub); dim(testingSub) #trainingSub will have 1 more variable - classe


# validating that all columns have all values... so no columns need imputing. Phew.
colSums(is.na(trainingSub)>0)



## plot any scatterplot matrix to see co-related variables and do PCA
p1 <- qplot(log(roll_belt), data = myTraining, binwidth=2, fill = classe, show.legend = FALSE)
p2 <- qplot(log(magnet_dumbbell_y), data = trainingSub, binwidth=2, fill = classe,show.legend = FALSE)
p3 <- qplot(log(pitch_forearm), data = trainingSub,binwidth=2,  fill = classe,show.legend = FALSE)
p4 <- qplot(log(accel_forearm_x), data = trainingSub, binwidth=2, fill = classe)

grid.arrange(p1, p2, p3, p4, nrow = 1)


## Principal component analysis begings
prin_train <- prcomp(trainingSub[keepCols], scale. = T)
std_dev <- prin_train$sdev
pr_var <- std_dev^2
prop_varex <- pr_var/sum(pr_var)
# Simple plot b/w Principal Components and Variance.
plot(cumsum(prop_varex), xlab = "Principal Component", 
     ylab = "Cumulative Proportion of Variance Explained", type = "b")
### so around 20 or 25 variables are important


#recursive feature selections
#control <- rfeControl(functions = rfFuncs,
#                      method = "repeatedcv",
#                      repeats = 3,
#                      verbose = FALSE)
#outcomeName<- 'classe'
#predictors<-names(trainingSub)[!names(trainingSub) %in% outcomeName]
#feature_Profile <- rfe(trainingSub[,predictors], trainingSub[,outcomeName],rfeControl = control)
#feature_Profile


######################################################
### begins the model building
######################################################
# decision tree
system.time(model_dt <- rpart(classe ~ ., data = trainingSub, method="class"))
fancyRpartPlot(model_dt)   ##Graph 2
prediction <- predict(model_dt, testingSub, type = "class")
confusionMatrix(prediction, myTesting$classe)
 
##  72% accuracy

# random forest
control <- trainControl(method = "repeatedcv", repeats = 3,verboseIter=FALSE, number=10, allowParallel = TRUE, search="random")
system.time(model_df <- train(classe ~.,method="rf",ntree = 100, data=trainingSub,trControl = control,verbose = FALSE))
ggplot(model_df)
model_df$finalModel
plot(varImp(object=model_df),main="RF - Variable Importance")
prediction_df <- predict(model_df, testingSub, type = "raw")
confusionMatrix(prediction_df, myTesting$classe)

#gbm
control <- trainControl(method = "repeatedcv", repeats = 3,verboseIter=FALSE, number=5, allowParallel = TRUE)
system.time(model_gbm <- train(classe ~.,method="gbm",trControl = control,data=trainingSub,  verbose = FALSE))
ggplot(model_gbm)
model_gbm$finalModel
plot(varImp(object=model_gbm),main="GBM - Variable Importance")
prediction_gbm <- predict(model_gbm, testingSub, type = "raw")
confusionMatrix(prediction_gbm, myTesting$classe)



final_pred <- predict(model_df, testing[keepCols], type = "raw")



