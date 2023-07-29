install.packages("RCurl")
install.packages("GGally")
install.packages("xgboost")

library(RCurl)
library(dplyr)
library(caret)
library(PerformanceAnalytics)
library(gridExtra)
library(ggplot2)
library(GGally)
library(xgboost)
library(pROC)

# UCI_data_URL2 <- getURL('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data')
UCI_data_URL <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
BC_wiscosin <- read.csv(UCI_data_URL, header = FALSE)
str(BC_wiscosin)
colnames(BC_wiscosin) <- c('id_number', 'diagnosis', 'radius_mean', 
                           'texture_mean', 'perimeter_mean', 'area_mean', 
                           'smoothness_mean', 'compactness_mean', 
                           'concavity_mean','concave_points_mean', 
                           'symmetry_mean', 'fractal_dimension_mean',
                           'radius_se', 'texture_se', 'perimeter_se', 
                           'area_se', 'smoothness_se', 'compactness_se', 
                           'concavity_se', 'concave_points_se', 
                           'symmetry_se', 'fractal_dimension_se', 
                           'radius_worst', 'texture_worst', 
                           'perimeter_worst', 'area_worst', 
                           'smoothness_worst', 'compactness_worst', 
                           'concavity_worst', 'concave_points_worst', 
                           'symmetry_worst', 'fractal_dimension_worst')

str(BC_wiscosin)
dim(BC_wiscosin)

BC_wiscosin <- subset(BC_wiscosin, select=-c(id_number)) # removes the id attribute from dataset
BC_wiscosin_eda <- BC_wiscosin
BC_wiscosin_eda$diagnosis<-ifelse(BC_wiscosin_eda$diagnosis == "B", 0, 1)

BC_wiscosin_mean = cbind(diagnosis=BC_wiscosin_eda[,c(1)], BC_wiscosin_eda[,c(2:11)])
BC_wiscosin_se = cbind(diagnosis=BC_wiscosin_eda[,c(1)], BC_wiscosin_eda[,c(12:21)])
BC_wiscosin_worst = cbind(diagnosis=BC_wiscosin_eda[,c(1)], BC_wiscosin_eda[,c(22:31)])
par(mfrow = c(3,1))
b1 <- boxplot(BC_wiscosin_mean, las=2, col="green", main="Breast Cancer Box-Plot for Mean", ylim = c(0,150))
b2 <- boxplot(BC_wiscosin_se, las=2, col="green", main="Breast Cancer Box-Plot for SE", ylim = c(0,150))
b3 <- boxplot(BC_wiscosin_worst, las=2, col="green", main="Breast Cancer Box-Plot for Worst", ylim = c(0,150))

chart.Correlation(BC_wiscosin_mean,histogram=TRUE,pch=19, main="Chart of Cancer Means ")
chart.Correlation(BC_wiscosin_se,histogram=TRUE,pch=19, main="Chart of Cancer Se ")
chart.Correlation(BC_wiscosin_worst,histogram=TRUE,pch=19, main="Chart of Cancer Worst ")

ggcorr(BC_wiscosin_eda, nbreaks=8, palette='PRGn', label=TRUE, 
       label_size=2, size = 1.8, label_color='black') + 
  ggtitle("Breast Cancer Correlation Matrix") + 
  theme(plot.title = element_text(hjust = 0.5, color = "grey15"))


#data analysis
table(BC_wiscosin$diagnosis)
round(prop.table(table(BC_wiscosin$diagnosis)) * 100, digits = 1)

# set.seed(1123)
## splitting the data
set.seed(1123)
trainIndex <- createDataPartition(BC_wiscosin$diagnosis, p = .8, list = FALSE, times = 1)
training_set <- BC_wiscosin[ trainIndex, ]
test_set <- BC_wiscosin[ -trainIndex, ]


p1 <- ggplot(BC_wiscosin, aes(x = diagnosis)) +
  geom_bar(aes(fill = "blue")) +
  ggtitle("Breast Cancer diagnosis for the entire dataset") +
  theme(legend.position="none", panel.background = element_rect(fill="lightblue"))

p2 <- ggplot(training_set, aes(x = diagnosis)) + 
  geom_bar(aes(fill = 'blue')) + 
  ggtitle("Breast Cancer diagnosis for the training data") + 
  theme(legend.position="none", panel.background = element_rect(fill="lightblue"))

p3 <- ggplot(test_set, aes(x = diagnosis)) + 
  geom_bar(aes(fill = 'blue')) + 
  ggtitle("Breast Cancer diagnosis for the testing data") + 
  theme(legend.position="none", panel.background = element_rect(fill="lightblue"))

grid.arrange(p1,p2,p3, ncol=3, nrow=1)


####techniques

fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",number = 3, repeats = 10) ## repeated ten times


#decision trees

dtree_model <- train(as.factor(diagnosis) ~ ., 
                     data = training_set, 
                     method = "rpart", 
                     metric = "Accuracy", 
                     trControl = fitControl)

feature_importance <- varImp(dtree_model, scale = FALSE)
feature_importance_scores <- data.frame(feature_importance$importance)

feature_importance_scores <- data.frame(names = row.names(feature_importance_scores), 
                                        var_imp_scores = feature_importance_scores$Overall)

predict_values <- predict(dtree_model,newdata = test_set)
confusionMatrix(as.factor(test_set$diagnosis),predict_values)

predict_values <- predict(dtree_model, newdata = training_set)
confusionMatrix(as.factor(training_set$diagnosis),predict_values)

ggplot(feature_importance_scores, 
       aes(reorder(names, var_imp_scores), var_imp_scores)) + 
  geom_bar(stat='identity', 
           fill = '#875FDB') + 
  theme(panel.background = element_rect(fill = '#fafafa')) + 
  coord_flip() + 
  labs(x = 'Feature', y = 'Importance') + 
  ggtitle('Importance of specific feature for decision trees')

predict_values <- predict(dtree_model,newdata = test_set)
confusionMatrix(as.factor(test_set$diagnosis),predict_values)

predict_values <- predict(dtree_model, newdata = training_set)
confusionMatrix(as.factor(training_set$diagnosis),predict_values)




#logistic regression

LR_model <- train(diagnosis ~ ., 
                  data = training_set, 
                  method = "glmnet",
                  metric = "Accuracy", 
                  family="binomial",
                  trControl = fitControl)

feature_importance1 <- varImp(LR_model, scale = FALSE)
feature_importance_scores1 <- data.frame(feature_importance1$importance)

feature_importance_scores1 <- data.frame(names = row.names(feature_importance_scores1), 
                                         var_imp_scores1 = feature_importance_scores1$Overall)

ggplot(feature_importance_scores1, 
       aes(reorder(names, var_imp_scores1), var_imp_scores1)) + 
  geom_bar(stat='identity', 
           fill = '#875FDB') + 
  theme(panel.background = element_rect(fill = '#fafafa')) + 
  coord_flip() + 
  labs(x = 'Feature', y = 'Importance') + 
  ggtitle('Importance of specific feature for logistic regression')

predict_values <- predict(LR_model,newdata = test_set)
confusionMatrix(as.factor(test_set$diagnosis),predict_values)
predict_values <- predict(LR_model, newdata = training_set)
confusionMatrix(as.factor(training_set$diagnosis),predict_values)



#Support Vector Machine
training_set_svm <- training_set
training_set_svm$diagnosis <- as.factor(training_set_svm$diagnosis)
char_columns <- sapply(training_set_svm, is.character)
training_set_svm[ , char_columns] <- as.data.frame(apply(training_set_svm[ , char_columns], 2, as.numeric))


svm_model <- train(diagnosis ~ ., 
                   data = training_set_svm, 
                   method = "svmLinear",
                   metric = "Accuracy", 
                   trControl = fitControl)

feature_importance2 <- varImp(svm_model, scale = FALSE)

# plot(feature_importance2)
ggplot(feature_importance2, 
       aes(reorder(names, Importance), Importance)) + 
  geom_bar(stat='identity', 
           fill = '#875FDB') + 
  theme(panel.background = element_rect(fill = '#fafafa')) + 
  coord_flip() + 
  labs(x = 'Feature', y = 'Importance') +
  ggtitle('Feature Importance for support vector machines')

predict_values <- predict(svm_model, newdata = test_set)
confusionMatrix(as.factor(test_set$diagnosis),predict_values)
predict_values <- predict(svm_model, newdata = training_set_svm)
confusionMatrix(as.factor(training_set_svm$diagnosis),predict_values)


## Gradient Boosting Machine model

# set.seed(1123)
set.seed(3011)
trainIndex1 <- createDataPartition(BC_wiscosin$diagnosis, p = .8, list = FALSE, times = 1)
train_all <- BC_wiscosin[ trainIndex1, ]
test_all <- BC_wiscosin[ -trainIndex1, ]

## Creating the independent variable and label matricies of train/test data
train_all_data  <- as.matrix(train_all[-1])
train_all_label <- train_all$diagnosis
## Converting labels to 0,1 where "M" is coded at 1
train_all_label <- as.numeric(c("M" = "1", "B" = "0")[train_all_label])
train_all$diagnosis[1:5]; train_all_label[1:5]

## Repeat for test dataset
test_all_data   <- as.matrix(test_all[-1])
test_all_label  <- test_all$diagnosis
test_all_label <- as.numeric(c("M" = "1", "B" = "0")[test_all_label])
test_all$diagnosis[1:5]; test_all_label[1:5]


train_all_data <- as.data.frame(apply(train_all_data, 2, as.numeric))
test_all_data <- as.data.frame(apply(test_all_data, 2, as.numeric))

## Formatting data for XGBoost matricies
all_dtrain = xgb.DMatrix(data = as.matrix(sapply(train_all_data,as.numeric)), label=as.matrix(train_all_label))
all_dtest = xgb.DMatrix(data = as.matrix(sapply(test_all_data,as.numeric)), label=as.matrix(test_all_label))


### parameters: max_depth, eta, subsample, colsample_bytree, and min_child_weight
all_low_err_list <- list()
all_parameters_list <- list()
set.seed(99)
for(i in 1:100){
  params <- list(booster = "gbtree",
                 objective = "binary:logistic",
                 max_depth = sample(3:25, 1),
                 eta = runif(1, 0.01, 0.3),
                 subsample = runif(1, 0.5, 1),
                 colsample_bytree = runif(1, 0.5, 1),
                 min_child_weight = sample(0:10, 1)
  )
  
  parameters <- as.data.frame(params)
  all_parameters_list[[i]] <- parameters
}

all_parameters_df <- do.call(rbind, all_parameters_list) #df containing random search params

### Fitting xgboost models based on search parameters
for (row in 1:nrow(all_parameters_df)){
  set.seed(99)
  all_tmp_mdl <- xgb.cv(data = all_dtrain,
                        booster = "gbtree",
                        objective = "binary:logistic",
                        nfold = 5,
                        prediction = TRUE,
                        max_depth = all_parameters_df$max_depth[row],
                        eta = all_parameters_df$eta[row],
                        subsample = all_parameters_df$subsample[row],
                        colsample_bytree = all_parameters_df$colsample_bytree[row],
                        min_child_weight = all_parameters_df$min_child_weight[row],
                        nrounds = 200,
                        eval_metric = "error",
                        early_stopping_rounds = 20,
                        print_every_n = 500,
                        verbose = 0
  )
  
  #this is the lowest error for the iteration
  all_low_err <- as.data.frame(1 - min(all_tmp_mdl$evaluation_log$test_error_mean))
  all_low_err_list[[row]] <- all_low_err
}

all_low_err_df <- do.call(rbind, all_low_err_list) #accuracies 
all_randsearch <- cbind(all_low_err_df, all_parameters_df) #data frame with everything

###Reformatting the dataframe
all_randsearch <- all_randsearch %>%
  dplyr::rename(val_acc = '1 - min(all_tmp_mdl$evaluation_log$test_error_mean)') %>%
  dplyr::arrange(-val_acc)

###Grabbing just the top model
all_randsearch_best <- all_randsearch[1,]

###Storing best parameters in list
all_best_params <- list(booster = all_randsearch_best$booster,
                        objective = all_randsearch_best$objective,
                        max_depth = all_randsearch_best$max_depth,
                        eta = all_randsearch_best$eta,
                        subsample = all_randsearch_best$subsample,
                        colsample_bytree = all_randsearch_best$colsample_bytree,
                        min_child_weight = all_randsearch_best$min_child_weight)


### Finding the best nround parameter for the model using 5-fold cross validation
set.seed(99)
all_xgbcv <- xgb.cv(params = all_best_params,
                    data = all_dtrain,
                    nrounds = 500,
                    nfold = 5,
                    prediction = TRUE,
                    print_every_n = 50,
                    early_stopping_rounds = 25,
                    eval_metric = "error",
                    verbose = 0
)
all_xgbcv$best_iteration


## Model training using best hyperparameters
set.seed(99)
all_best_xgb <- xgb.train(params = all_best_params,
                          data = all_dtrain,
                          nrounds = all_xgbcv$best_iteration,
                          eval_metric = "error",
)

xgb.save(all_best_xgb, 'final_xgb_cancerall')


## Model testing and visualizations
cancer_all.pred <- predict(all_best_xgb, all_dtest)
cancer_all.pred <- factor(ifelse(cancer_all.pred > 0.5, 1, 0),
                          labels = c("B", "M"))
test_all$diagnosis <- as.factor(test_all$diagnosis)
confusionMatrix(cancer_all.pred, test_all$diagnosis,
                mode = 'everything',
                positive = 'M')

cancer_all_train.pred <- predict(all_best_xgb, all_dtrain)
cancer_all_train.pred <- factor(ifelse(cancer_all_train.pred > 0.5, 1, 0),
                          labels = c("B", "M"))
train_all$diagnosis <- as.factor(train_all$diagnosis)
confusionMatrix(cancer_all_train.pred, train_all$diagnosis,
                mode = 'everything',
                positive = 'M')

## Visualizations
all_impt_mtx <- xgb.importance(feature_names = colnames(test_all_data), model = all_best_xgb)
xgb.plot.importance(importance_matrix = all_impt_mtx,
                    xlab = "Variable Importance")


### ROC curve for 5-fold CV random parameter search
all_randsearch_roc <- roc(response = train_all_label,
                          predictor = all_tmp_mdl$pred,
                          print.auc = TRUE,
                          plot = TRUE)

### ROC curve for 5-fold CV nround parameter search
all_nround_roc <- roc(response = train_all_label,
                      predictor = all_xgbcv$pred,
                      print.auc = TRUE,
                      plot = TRUE)

