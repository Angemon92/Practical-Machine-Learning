Practical Machine Learning Course Project Report
===============================================
`Stefan Jovanovic 92'`   
`27.10.2014.`




```r
library(caret)
library(rpart)
library(randomForest)
```

### Load data

Download data, and load into *your working directory*.

```r
data = read.csv("pml-training.csv")

# For second part of project.
readTest = read.csv("pml-training.csv") 
```
Create training (70%) and test (30%) set. 

```r
inTrain <- createDataPartition(data$classe, p=0.7, list=FALSE)
train <- data[inTrain,]
test <- data[-inTrain,]
```

### Clean the data. 

1.Remove attributes with low variance.

```r
nzv = nearZeroVar(train,saveMetric=TRUE)
train = train[,!nzv$nzv]
```

2.Remove attributes with more than 75% missing values.

```r
nav <- sapply(colnames(train), function(x) if( sum(is.na(train[, x])) > 0.75*nrow(train) )
                                           {return(T)}else{return(F)})
train <- train[, !nav]
```

3.Calculate correlations (each variable vs class column) and find highest.

```r
# Remove column with row numbers.
train = train[,-1]        

corSpearman <- abs(sapply(colnames(train[, -ncol(train)]), function(x) 
                                                cor(as.numeric(train[, x]), 
                                                as.numeric(train$classe),
                                                method ="spearman")))
summary(corSpearman)
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##  0.0015  0.0147  0.0524  0.0862  0.1370  0.3170
```

4.Plot two most affecting predictors.

```r
plot(train[, names(which.max(corSpearman))], train$classe,col=train$classe,
     pch = 19, cex = 0.1,
     xlab = names(which.max(corSpearman)),
     ylab = "Classes" )
```

![plot of chunk cleaning4](figure/cleaning41.png) 

```r
plot(train[, names(which.max(corSpearman[-which.max(corSpearman)]))], train$classe,col=train$classe,
     pch = 19, cex = 0.1,
     xlab = names(which.max(corSpearman[-which.max(corSpearman)])),
     ylab = "Classes" )
```

![plot of chunk cleaning4](figure/cleaning42.png) 

### First phase conclusion:
There doesn't seem to be any strong predictors that correlates with `classe` well, so linear regression will not give as a good model for predicting. Lets try with **trees**.

#### Classification tree, with cross validation.

```r
modelFit <- train(classe ~.,method="rpart",data=train,
                  trControl = trainControl(method = "cv", number = 10))
modelFit
```

```
## CART 
## 
## 19622 samples
##    57 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 17659, 17659, 17660, 17661, 17659, 17659, ... 
## 
## Resampling results across tuning parameters:
## 
##   cp    Accuracy  Kappa  Accuracy SD  Kappa SD
##   0.04  0.5       0.35   0.07         0.12    
##   0.05  0.4       0.22   0.07         0.13    
##   0.12  0.3       0.06   0.04         0.06    
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.04024.
```
Confusion matrix for this model is:

```r
confusionMatrix( table( predict(modelFit,test), test$classe ) )
```

```
## Confusion Matrix and Statistics
## 
##    
##        A    B    C    D    E
##   A 4243  394    6    0    0
##   B  894 2128  515 1115  288
##   C  429 1275 2901 2101 1688
##   D    0    0    0    0    0
##   E   14    0    0    0 1631
## 
## Overall Statistics
##                                         
##                Accuracy : 0.556         
##                  95% CI : (0.549, 0.563)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.44          
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.760    0.560    0.848    0.000   0.4522
## Specificity             0.972    0.822    0.661    1.000   0.9991
## Pos Pred Value          0.914    0.431    0.346      NaN   0.9915
## Neg Pred Value          0.911    0.886    0.954    0.836   0.8901
## Prevalence              0.284    0.194    0.174    0.164   0.1838
## Detection Rate          0.216    0.108    0.148    0.000   0.0831
## Detection Prevalence    0.237    0.252    0.428    0.000   0.0838
## Balanced Accuracy       0.866    0.691    0.754    0.500   0.7257
```
Poor accuracy...

#### Random forest.

```r
modelFitRF = randomForest(classe ~ . , train)
modelFitRF
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = train) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.06%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 5579    1    0    0    0   0.0001792
## B    2 3795    0    0    0   0.0005267
## C    0    3 3418    1    0   0.0011689
## D    0    0    2 3213    1   0.0009328
## E    0    0    0    1 3606   0.0002772
```
Confusion matrix for this model is:

```r
confusionMatrix( table( predict(modelFitRF,test), test$classe) )
```

```
## Confusion Matrix and Statistics
## 
##    
##        A    B    C    D    E
##   A 1674    0    0    0    0
##   B    0 1139    0    0    0
##   C    0    0 1026    0    0
##   D    0    0    0  964    0
##   E    0    0    0    0 1082
## 
## Overall Statistics
##                                     
##                Accuracy : 1         
##                  95% CI : (0.999, 1)
##     No Information Rate : 0.284     
##     P-Value [Acc > NIR] : <2e-16    
##                                     
##                   Kappa : 1         
##  Mcnemar's Test P-Value : NA        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```
### Good metrics result, so this is the final model im going to chose.


## Project part 2.
Load testData, do predictions, and save each prediction in .txt file

```r
# Submitting function (from Cours)
        pml_write_files = function(x){
                n = length(x)
                for(i in 1:n){
                        filename = paste0("predictions/problem_id_",i,".txt")
                        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
                }
        }

answers = predict(modelFitRF,readTest)
pml_write_files(answers)
```
