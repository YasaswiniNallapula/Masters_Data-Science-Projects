#installing libraries
library(caret)
install.packages("DataExplorer")
library(DataExplorer)
library(dplyr)
library(glue)
library(randomForest)
library(gmodels)

#Load the dataset into R
setwd("Path_to_location_of_file")
getwd()
bank <- read.csv("bank_marketing.csv", header=TRUE, sep=",")

#structure of dataset
str(bank)
dim(bank)

#summary of dataset
summary(bank)


#Data Validation
##check for duplicate rows
sum(duplicated(bank))
##check for rows with Missing values 
sum(!complete.cases(bank))
##check for Missing values by variable
sapply(bank, function(x) sum(is.na(x)))


#convert character to factor type
bank <- as.data.frame(unclass(bank), stringsAsFactors = TRUE)

#print all the levels of factor variables
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

#univariate analysis

plot(bank$education, col = "magenta", main = "Education distribution of customers")
plot(bank$job, col = "cyan", main = "Job distribution of customers")

par(mfrow = c(3, 3))

p1 <- hist(bank$age, col = "green", main = "Age distribution of customers")
p4 <- plot(bank$marital, col = "green", main = "marital status of customers")
p5 <- hist(bank$balance, col = "magenta", xlim = c(0,20000), main = "Histogram of bank$balance")
p6 <- hist(bank$campaign, col = "cyan", xlim = c(0,25), main = "Histogram of bank$campaign")
p7 <- hist(bank$day, col = "green", main = "Histogram of bank$day")
p8 <- hist(bank$duration, col = "magenta", xlim = c(0,2000), main = "Histogram of bank$duration")
p10 <- hist(bank$previous, col = "green", xlim = c(0,40), main = "Histogram of bank$previous")
p11 <- plot(bank$loan, col = "magenta", main = "Plot of bank$personal loan")
p12 <- plot(bank$housing, col = "cyan", main = "Plot of bank$housing loan")



# Multi variate analysis

age_group <- bank %>% group_by(y) %>% summarize(grp.mean=mean(age))
ggplot (bank, aes(x=age)) + 
  geom_histogram(color = "yellow", fill = "blue", binwidth = 5) +
  facet_grid(cols=vars(y)) + 
  ggtitle('Age Distribution by Subscription to Term deposit') + ylab('Count') + xlab('Age') +
  scale_x_continuous(breaks = seq(0,100,5))

duration_group <- bank %>% group_by(y) %>% summarize(grp.mean=mean(duration))
ggplot (bank, aes(x=duration)) + 
  geom_histogram(color = "blue", fill = "blue", binwidth = 5) +
  facet_grid(cols=vars(y)) + 
  ggtitle('Duration Distribution by subscription to Term deposit') + ylab('Count') + xlab('Age') +
  scale_x_continuous(breaks = seq(0,3200,400))

balance_group <- bank %>% group_by(y) %>% summarize(grp.mean=mean(balance))
ggplot (bank, aes(x=balance)) + 
  geom_histogram(color = "blue", fill = "blue") +
  facet_grid(cols=vars(y)) + 
  ggtitle('Balance Loan Distribution by subscription to Term deposit') + ylab('Count') + xlab('Balance') +
  geom_vline(data=balance_group, aes(xintercept=grp.mean), color="red", linetype="dashed")

ggplot(data=bank, aes(x=campaign, fill=y))+
  geom_histogram()+
  ggtitle("Subscription based on Number of Contact during the Campaign")+
  xlab("Number of Contact during the Campaign")+
  xlim(c(min=1,max=30)) +
  guides(fill=guide_legend(title="Subscription of Term Deposit"))

barplot(table(bank$y, bank$education), main="Distribution of Education vs Term Deposit",
        xlab="Education", col=c("darkblue","red"),
        legend = rownames(table(bank$y, bank$education)), beside=TRUE)

barplot(table(bank$y, bank$job), main="Distribution of Job vs Term Deposit",
        xlab="Job", col=c("darkblue","red"),
        legend = rownames(table(bank$y, bank$job)), beside=TRUE)

barplot(table(bank$y, bank$marital), main="Distribution of Marital status vs Deposit",
        xlab="Marital status", col=c("darkblue","red"),
        legend = rownames(table(bank$y, bank$marital)), beside=TRUE)

barplot(table(bank$y, bank$month), main="Distribution of Last contact month vs Deposit",
        xlab="Month", col=c("darkblue","red"),
        legend = rownames(table(bank$y, bank$month)), beside=TRUE)


#split the data into training and testing for the Random Forest
bank_rf <- bank
set.seed(126) # Set random seed for reproducibility
train_indices_rf <- createDataPartition(bank_rf$y, p = 0.85, list = FALSE)
bank_rf_train <- bank_rf[train_indices_rf, ]
bank_rf_test <- bank_rf[-train_indices_rf, ]

# separate the test labels from the test data
bank_rf_test_data <- bank_rf_test[1:16]
bank_rf_test_label <- bank_rf_test[,17]
str(bank_rf_test_data)
str(bank_rf_test_label)

# Creating a random forest classifier
rf_model <- randomForest(formula = y ~ ., data = bank_rf_train)
rf_model

# Making predictions on the testing data
deposit_pred <- predict(rf_model, bank_rf_test_data)


# Evaluating the accuracy of the model
accuracy <- confusionMatrix(deposit_pred, bank_rf_test_label)$overall["Accuracy"]
print(paste("Accuracy:", round(accuracy, 4)))

#comparing the predicted and actual values
CrossTable(x=bank_rf_test_label, y=deposit_pred, prop.chisq=FALSE)
confusionMatrix(deposit_pred, bank_rf_test_label)


# variable/feature importance
varImpPlot(rf_model)






