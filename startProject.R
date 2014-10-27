go <- function(){
        library(caret)
        
        readTrain = read.csv("pml-training.csv")
        readTest = read.csv("pml-testing.csv")
        
        inTrain <- createDataPartition(readTrain$classe, p=0.7, list=FALSE)
        train <- readTrain[inTrain,]
        test <- readTrain[-inTrain,]
        rm(readTrain, inTrain)
        
# remove attributes with low variance        
        nzv = nearZeroVar(train,saveMetric=TRUE)
        train = train[,!nzv$nzv]
        rm(nzv)

# remove attributes with more than 75% missing values
        nav <- sapply(colnames(train), function(x)
                                                if( sum(is.na(train[, x])) > 0.75*nrow(train) )
                                                        {return(T)}else{return(F)})
        train <- train[, !nav]
        rm(nav)

# calculate correlations (with class column) and find highest
        # Remove row numbers
        train = train[,-1]        

        corSpearman <- abs(sapply(colnames(train[, -ncol(train)]), function(x)
                                                                cor(as.numeric(train[, x]), as.numeric(train$classe), method = "spearman")))
        summary(corSpearman)

        
# plot two most affecting predictors
        plot(train[, names(which.max(corSpearman))], train$classe,col=train$classe ,pch = 19, cex = 0.1, xlab = names(which.max(corSpearman)), ylab = "Classes" )
        plot(train[, names(which.max(corSpearman[-which.max(corSpearman)]))], train$classe,col=train$classe, pch = 19, cex = 0.1, xlab = names(which.max(corSpearman)),  ylab = "Classes" )


#  First try a classification tree
       fitControl <- trainControl(method = "cv", number = 10)
        modelFit <- train(classe ~.,method="rpart",data=train), trControl = fitControl)

        plot(modelFit)

        plot(modelFit$finalModel)
        text(modelFit$finalModel, use.n=F, all=T, cex=.7)
        modelFit

        confusionMatrix( table( predict(modelFit,test), test$classe ) ) #A=5557

# Random forests model
        modelFitRF = randomForest(classe ~ . , train)
        modelFitRF

        confusionMatrix( table( predict(modelFitRF2,test), test$classe ) )

# Submitting function (from Cours)
        pml_write_files = function(x){
                n = length(x)
                for(i in 1:n){
                        filename = paste0("predictions/problem_id_",i,".txt")
                        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
                }
        }


# Submit
answers = predict(modelFitRF,readTest)
pml_write_files(answers)

}