
#installing libraries
library(caret)
library(caTools)
install.packages("DataExplorer")
library(DataExplorer)
install.packages("ROCR")
library(ROCR)
library(dplyr)
library(glue)
#install.packages("randomForest")
library(e1071)
library(randomForest)
#library(partykit)
#library(rpart.plot)
#install.packages("naivebayes")
#library(naivebayes)

#Load the dataset into R
bank <- read.csv("Path_to_local_directory/bank_marketing.csv",header=TRUE,sep=",")
colnames(bank) = c("age", "job", "marital", "education", "default", "balance", "housing", "loan", 
                   "contact", "day", "month", "duration", "campaign", "pdays", "previous", 
                   "poutcome", "y")

#description of dataset
summary(bank)
str(bank)
dim(bank)


#Data Validation
##check for duplicate rows
sum(duplicated(bank))
##check for Missing values
sapply(bank, function(x) sum(is.na(x)))
plot_missing(bank, title='plot for missing values', ggtheme=theme_linedraw(),
             theme_config=list(legend.position=c("bottom")))

#convert character to factor type
bank <- as.data.frame(unclass(bank), stringsAsFactors = TRUE)
bank_lr <- bank
#convert int to factor type
bank$y <- factor(bank$y, levels = c(1,2), labels = c('no', 'yes'))
bank_lr$y <- ifelse(bank_lr$y==2, 1,0)
#print all the levels of factor variables to find any missing values
levels(bank$job)
levels(bank$marital)
levels(bank$education)
levels(bank$default)
levels(bank$housing)
levels(bank$loan)
levels(bank$contact)
levels(bank$month)
levels(bank$poutcome)
levels(bank$y)

#data analysis
table(bank$y)
round(prop.table(table(bank$y)) * 100, digits = 1)


job_distribution <- bank %>% group_by(job) %>% summarise(job_count = n()) %>% arrange(-job_count)
job_distribution_plot <- ggplot(data = job_distribution, aes(x = job_count, 
                                                             y = reorder(job, job_count), 
                                                             text = glue("No. of customers: {job_count}")
)) +
  geom_col(aes(fill = job)) +
  labs( title = "Job Distribution of Customers",
        x = "No. of Jobs",
        y = "jobs"
  ) +
  theme_minimal() +
  theme(legend.position = "none")
job_distribution_plot

education_distribution <- bank %>% group_by(education) %>% summarise(ed_count = n())
education_distribution_plot <- ggplot(data = education_distribution, aes(y = ed_count, 
                                                                         x = education, 
                                                                         text = glue("No. of customers: {ed_count}")
)) +
  geom_col(aes(fill = "brickred")) +
  labs( title = "Education Distribution of Customers",
        x = "Education Level",
        y = ""
  ) +
  theme_minimal() +
  theme(legend.position = "none")
education_distribution_plot


mu <- bank %>% group_by(y) %>% summarize(grp.mean=mean(age))
ggplot (bank, aes(x=age)) + 
  geom_histogram(color = "blue", fill = "blue", binwidth = 5) +
  facet_grid(cols=vars(y)) + 
  ggtitle('Age Distribution by Subscription') + ylab('Count') + xlab('Age') +
  scale_x_continuous(breaks = seq(0,100,5)) +
  geom_vline(data=mu, aes(xintercept=grp.mean), color="red", linetype="dashed")

ggplot(data = bank, aes(x=education, fill=y)) +
  geom_bar() +
  ggtitle("Term Deposit Subscription based on Education Level") +
  xlab(" Education Level") +
  guides(fill=guide_legend(title="Subscription of Term Deposit"))

ggplot(data = bank, aes(x=job, fill=y)) +
  geom_bar() +
  ggtitle("Term Deposit Subscription based on job") +
  xlab(" Job") +
  guides(fill=guide_legend(title="Subscription of Term Deposit"))


#transforming the numerical variables using scale
bank_lr[c(1,6,10,12,13)] <- scale(bank_lr[c(1,6,10,12,13)])
#split the dataset into training and testing for logistic regression
set.seed(112)
split = sample.split(bank_lr$y,SplitRatio = 0.70)
bank_lr_training = subset(bank_lr, split == TRUE)
bank_lr_test = subset(bank_lr, split == FALSE)


#logistic regression model

binclass_eval = function (actual, predict) {
  cm = table(as.integer(actual), as.integer(predict), dnn=c('Actual','Predicted'))
  ac = (cm['1','1']+cm['0','0'])/(cm['0','1'] + cm['1','0'] + cm['1','1'] + cm['0','0'])
  pr = cm['1','1']/(cm['0','1'] + cm['1','1'])
  rc = cm['1','1']/(cm['1','0'] + cm['1','1'])
  fs = 2* pr*rc/(pr+rc)
  list(cm=cm, recall=rc, precision=pr, fscore=fs, accuracy=ac)
}


plot_pred_type_distribution <- function(df, threshold) {
  v <- rep(NA, nrow(df))
  v <- ifelse(df$pred >= threshold & df$y == 1, "TP", v)
  v <- ifelse(df$pred >= threshold & df$y == 0, "FP", v)
  v <- ifelse(df$pred < threshold & df$y == 1, "FN", v)
  v <- ifelse(df$pred < threshold & df$y == 0, "TN", v)
  
  df$pred_type <- v
  
  ggplot(data=df, aes(x=y, y=pred)) + 
    geom_violin(fill='black', color=NA) + 
    geom_jitter(aes(color=pred_type), alpha=0.6) +
    geom_hline(yintercept=threshold, color="red", alpha=0.6) +
    scale_color_discrete(name = "type") +
    labs(title=sprintf("Threshold at %.2f", threshold))
}

#creating the LR classifer
classifier.lm = glm(formula = y ~ .,
                    family = binomial,
                    data = bank_lr_training)

#Evaluating the LR model
pred_lm = predict(classifier.lm, type='response', newdata=bank_lr_test[-17])
predictions_LR <- data.frame(y = bank_lr_test$y, pred = NA)
predictions_LR$pred <- pred_lm
plot_pred_type_distribution(predictions_LR,0.30)

test.eval.LR = binclass_eval(bank_lr_test[, 17], pred_lm > 0.30)
test.eval.LR$cm
acc_LR=test.eval.LR$accuracy
prc_LR=test.eval.LR$precision
rc_LR=test.eval.LR$recall
cat("Accuracy:  ",   acc_LR,
    "\nPrecision: ", prc_LR,
    "\nRecall: ",rc_LR)


rocr.pred.lr = prediction(predictions = pred_lm, labels = bank_lr_test$y)
rocr.perf.lr = performance(rocr.pred.lr, measure = "tpr", x.measure = "fpr")
rocr.auc.lr = as.numeric(performance(rocr.pred.lr, "auc")@y.values)

rocr.auc.lr

plot(rocr.perf.lr,
     lwd = 3, colorize = TRUE,
     print.cutoffs.at = seq(0, 1, by = 0.1),
     text.adj = c(-0.2, 1.7),
     main = 'ROC Curve')
mtext(paste('Logistic Regression - auc : ', round(rocr.auc.lr, 5)))
abline(0, 1, col = "red", lty = 2)



# Split the data into training and testing sets for the Naive Bayes model
set.seed(1234)  # for reproducibility
trainIndex <- createDataPartition(bank$y, p = 0.7, list = FALSE)
bank_nb_train <- bank[trainIndex, ]
bank_nb_test <- bank[-trainIndex, ]

# Train the Naive Bayes model
nb_model <- naiveBayes(y ~ ., data = bank_nb_train)
nb_model

# Evaluate the model on the testing set
predictions <- predict(nb_model, newdata = bank_nb_test,na.action = na.pass)
confusionMatrix(predictions, bank_nb_test$y)



#split the data into training and testing for the Random Forest
bank_rf <- subset(bank, select = -c(duration))
set.seed(42) # Set random seed for reproducibility
train_indices_rf <- createDataPartition(bank_rf$y, p = 0.7, list = FALSE)
bank_rf_train <- bank_rf[train_indices_rf, ]
bank_rf_test <- bank_rf[-train_indices_rf, ]

# Creating a random forest classifier with 100 trees
rf_model <- randomForest(y ~ ., data = bank_rf_train, ntree = 100)
rf_model

# Making predictions on the testing data
y_pred <- predict(rf_model, bank_rf_test)

# Evaluating the accuracy of the model
accuracy <- confusionMatrix(y_pred, bank_rf_test$y)$overall["Accuracy"]
print(paste("Accuracy:", round(accuracy, 4)))






