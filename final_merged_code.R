library(ggplot2)
library(dplyr)
library(pastecs)
library(xgboost)
library(lubridate)
library(caret)
install.packages("imbalance")
install.packages("ROSE")
library(ROSE)
install.packages("DMwR")
library(smotefamily)
library(DMwR)
library(caret)
library(MASS)

raw_df <- read.csv("dataset/train_data.csv")
head(raw_df)

is.null(raw_df)

str(raw_df)

gender_count <- raw_df %>% 
  group_by(Gender) %>%
  summarise(count = n())

city_count <- raw_df %>%
  group_by(City) %>%
  summarise(count = n())

############ EDA Part 1 #########################
ggplot(gender_count, aes(x = Gender, y = count)) +
  geom_bar(stat = "identity", fill = "Blue")

ggplot(city_count, aes(x = City, y = count)) +
  geom_bar(stat = "identity", fill = "green")


att_emp_id <- raw_df[raw_df$LastWorkingDate != "", 2]



raw_df %>%
  mutate(
    label = if_else(Emp_ID %in% c(att_emp_id), 1, 0),
    color = if_else(Emp_ID %in% c(att_emp_id), "red", "yellow")
  ) %>%
  ggplot(aes(x = Age, y = Salary, colour = color)) +
  geom_point()

######### Data engineering ##############

att_emp <- raw_df %>%
  group_by(Emp_ID) %>%
  mutate(
    DaysBeforeDeparture = as.integer(
      as.Date(max(LastWorkingDate)) - as.Date(MMM.YY)
    )
  ) %>%
  mutate(
    differenceInSalary = Salary - lag(Salary)
  ) %>%
  mutate(
    days_worked = as.integer(
      as.Date(MMM.YY) - as.Date(Dateofjoining)
    )
  ) %>%
  mutate(
    is_promoted = ifelse(
      Designation - lag(Designation) == 0,
      0,
      1
    )
  )

# Remove NA from lag function with 0
att_emp$differenceInSalary[is.na(att_emp$differenceInSalary)] <- 0
att_emp$is_promoted[is.na(att_emp$is_promoted)] <- 0

att_emp1 = att_emp

#Factorize the data 
category_cars <- c("Gender", "Education_Level", "is_promoted",'LastWorkingDate')
att_emp1$Gender <- as.factor(att_emp1$Gender)
att_emp1$Education_Level <- as.factor(att_emp1$Education_Level)
att_emp1$is_promoted <- as.factor(att_emp1$is_promoted)
att_emp1$LastWorkingDate<-as.Date(att_emp1$LastWorkingDate)
empid = att_emp1$Emp_ID
str(att_emp1)
drop_cols <- c(
  "MMM.YY","Dateofjoining","City","Emp_ID"
)
new_att_emp1 <- att_emp1[, !names(att_emp1) %in% drop_cols]
gender_dummy <- model.matrix(~Gender - 1, data = new_att_emp1)
education_dummy <- model.matrix(~Education_Level - 1, data = new_att_emp1)
is_promoted_dummy <- model.matrix(~is_promoted - 1, data = new_att_emp1)

new_att_emp1 <- cbind(empid,new_att_emp1, gender_dummy,education_dummy,is_promoted_dummy)

new_att_emp1 <- new_att_emp1[, !names(new_att_emp1) %in% c("Gender", "Education_Level", "is_promoted")]
head(new_att_emp1)
str(new_att_emp1)

drop2 <-c('is_promoted0','Education_LevelMaster','GenderMale')
new_att_emp1 <- new_att_emp1[, !names(new_att_emp1) %in% drop2]
str(new_att_emp1)

#Finding out the employees that have left based on lastworkingday
new_att_emp1$LastWorkingDate <-as.character(new_att_emp1$LastWorkingDate)
new_att_emp1$LastWorkingDate[is.na(new_att_emp1$LastWorkingDate)] <- 0
new_att_emp1$LastWorkingDate[!new_att_emp1$LastWorkingDate=='0']<-1
new_att_emp1$LastWorkingDate = as.numeric(new_att_emp1$LastWorkingDate)
new_att_emp1$DaysBeforeDeparture[is.na(new_att_emp1$DaysBeforeDeparture)] <- 0

#Grouping employee information based on their ID
#(Since there are multiple entries of a particular employee)
grouped_data = new_att_emp1 %>%
  group_by(empid)
grouped_data_final = grouped_data %>% summarize(
  Salary = mean(Salary),
  Age = max(Age),
  JoiningDesignation = min(Joining.Designation),
  Designation = max(Designation),
  BusinessValue = mean(Total.Business.Value),
  QuaterlyRating = max(Quarterly.Rating),
  DiffInSalary = max(differenceInSalary),
  DaysWorked = max(days_worked),
  GenderFemale = max(GenderFemale),
  Education_LevelBachelor = max(Education_LevelBachelor),
  Education_LevelCollege = max(Education_LevelCollege),
  Ispromoted = max(is_promoted1),
  LWD = sum(LastWorkingDate),
  
  
)
grouped_data_final

#Generating test and train data 
set.seed(33)
index = createDataPartition(grouped_data_final$LWD,p=0.7,list=FALSE)
train_data = grouped_data_final[index,]
train_data = train_data[,-1]
test_data = grouped_data_final[-index,]
test_data = test_data[,-1]
table(grouped_data_final$LWD)


############ Logistic Regression without oversampling ############3
log_base_train = glm(LWD~.,train_data,family='binomial')
summary(log_base_train)
length(log_base_train$fitted.values)
nrow(train_data)
y_hat_train_class =ifelse(log_base_train$fitted.values < 0.5, 0, 1)
tab_train_class = table(
  y_hat_train_class,
  train_data$LWD,
  dnn = c("Predicted", "Actual")
)
#Confusion Matrix for train data
tab_train_class
confusionMatrix(tab_train_class)

y_hat_test_class = predict(log_base_train,test_data)
y_hat_test_glm <- as.numeric(y_hat_test_class > 0.5)
tab_test_class = table(y_hat_test_glm,test_data$LWD,dnn=c("Predicted","Actual"))
#Confusion Matrix for test data
confusionMatrix(tab_test_class)



######### Oversampling the data ##########

smote = SMOTE(train_data[,-13],train_data$LWD)
oversampled_train = smote$data
names(oversampled_train)[13] = "LWD"

######### Logistic Regression without oversampling ##########

oversampled_train$LWD = as.numeric(oversampled_train$LWD)
log_oversampled = glm(LWD~.,oversampled_train,family = 'binomial')
summary(log_oversampled)
table(oversampled_train$LWD)

y_hat_over_train_class =ifelse(log_oversampled$fitted.values < 0.5, 0, 1)
tab_over_train_class = table(
  y_hat_over_train_class,
  oversampled_train$LWD,
  dnn = c("Predicted", "Actual")
)
#Confusion Matrix for train data
tab_over_train_class
confusionMatrix(tab_over_train_class)

y_hat_over_test_class = predict(log_oversampled,test_data)
y_hat_over_test_glm <- as.numeric(y_hat_over_test_class > 0.5)
tab_over_test_class = table(y_hat_over_test_glm,test_data$LWD,dnn=c("Predicted","Actual"))
#Confusion Matrix for test data
confusionMatrix(tab_over_test_class)


######### Feature selection ##########

control = rfeControl(rfFuncs,"repeatedcv",2,5)
x_train = train_data[,-13]
y_train = train_data[,13]
x_test = test_data[,-13]
y_test = test_data[,13]
nrow(y_train)
stepAIC(log_base_train,direction = "backward")
fit2 = glm(LWD~1.,train_data,family='binomial')
stepAIC(fit2,direction='forward',scope=list(upper=log_base_train,lower=fit2))
stepAIC(fit2,direction='both',scope=list(upper=log_base_train,lower=fit2))






######### Extract left employees #########
att_emp <- att_emp %>%
  filter(Emp_ID %in% c(att_emp_id))



#### Plot the count of employees left in which month

att_emp %>%
  filter(LastWorkingDate != "") %>%
  ggplot(
    aes(
      x = month(as.Date(LastWorkingDate)))
  ) +
  geom_bar(fill = "purple") +
  scale_x_discrete(limit = c("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12")) + # nolint
  labs(x = "Months", y = "Number of departures", title = "Months v/s Departures") # nolint


# Create Factors for qualitative variable
category_cars <- c("Gender", "Education_Level", "is_promoted")
att_emp$Gender <- as.factor(att_emp$Gender)
att_emp$Education_Level <- as.factor(att_emp$Education_Level)
att_emp$is_promoted <- as.factor(att_emp$is_promoted)

str(att_emp)

drop_cols <- c(
  "MMM.YY",
  "Emp_ID",
  "City",
  "Dateofjoining",
  "LastWorkingDate",
  "Joining.Destination"
)

new_att_emp <- att_emp[, !names(att_emp) %in% drop_cols]

# Assign risk level to the employees
new_att_emp <- new_att_emp %>%
  mutate(
    riskLevel = if_else(
      DaysBeforeDeparture < 120,
      1,
      0
    )
  )

# Create dummy columns for categorical variables
gender_dummy <- model.matrix(~Gender - 1, data = new_att_emp)
education_dummy <- model.matrix(~Education_Level - 1, data = new_att_emp)
is_promoted_dummy <- model.matrix(~is_promoted - 1, data = new_att_emp)

new_att_emp <- cbind(new_att_emp, gender_dummy)
new_att_emp <- cbind(new_att_emp, education_dummy)
new_att_emp <- cbind(new_att_emp, is_promoted_dummy)

new_att_emp <- new_att_emp[, !names(new_att_emp) %in% c("Gender", "Education_Level", "is_promoted")]

# Check correlation of risk with other independent variables
new_att_emp$riskLevel <- as.integer(new_att_emp$riskLevel)
str(new_att_emp)
cor(new_att_emp)[, 10]

head(new_att_emp)
######### EDA Part 2 ##################
new_att_emp %>%
  ggplot(aes(x = riskLevel)) +
  geom_bar()

new_att_emp %>%
  ggplot(aes(x = DaysBeforeDeparture, y = Total.Business.Value)) +
  geom_point(aes(color = riskLevel)) +
  labs(title = "Relation between Business Value v/s Departure Date")

###########Removing Outliers#############

#Remove outliers from salary
boxplot(new_att_emp$Salary)
summary(new_att_emp$Salary)


new_att_emp <- new_att_emp %>%
  filter(Salary < 75835 & Salary > 38624)
table(new_att_emp$riskLevel)

# EDA for total business value
boxplot(new_att_emp$Total.Business.Value)
new_att_emp %>%
  ggplot(aes(x = Total.Business.Value)) +
  geom_histogram()

summary(new_att_emp$Total.Business.Value)
# Bringing business value rating in scale
new_att_emp$Total.Business.Value <- log(new_att_emp$Total.Business.Value + 2)
new_att_emp <- na.omit(new_att_emp)

####### Rating Distribution based on risk level ##############
risk_level0 <- new_att_emp %>%
  filter(riskLevel == 0)

risk_level0 %>%
  ggplot(aes(x = Quarterly.Rating)) +
  geom_bar()


risk_level1 <- new_att_emp %>%
  filter(riskLevel == 1)

risk_level1 %>%
  ggplot(aes(x = Quarterly.Rating)) +
  geom_bar()

####### Salary distribution based on risk level ##############
risk_level1 %>%
  ggplot(aes(x = Salary)) +
  geom_histogram() +
  labs(title = "Risky Employees current Salary")

risk_level0 %>%
  ggplot(aes(x = Salary)) +
  geom_histogram() +
  labs(title = "Non-Risky Employees Current Salary")


# Bringing the slary in scale and removing the skewedness

new_att_emp$Salary <- log(new_att_emp$Salary + 1)

train_sample <- sample(nrow(new_att_emp), 0.8 * nrow(new_att_emp))

att_emp_train <- new_att_emp[train_sample, ]

att_emp_test <- new_att_emp[-train_sample, ]

table(att_emp_train$riskLevel)
table(att_emp_test$riskLevel)

# check base model
base_model <- glm(riskLevel ~ ., att_emp_train, family = "binomial")

summary(base_model)

att_emp_train <- att_emp_train[, !names(att_emp_train) %in% 
                                 c("GenderMale", "Education_LevelMaster", "DaysBeforeDeparture", "is_promoted0", "is_promoted1")]

att_emp_test <- att_emp_test[,
                             !names(att_emp_test) %in% c(
                               "GenderMale", "Education_LevelMaster", "DaysBeforeDeparture", "is_promoted0", "is_promoted1"
                             )
]

head(att_emp_test)
test_data_x <- subset(att_emp_test, select = -c(riskLevel))
test_data_y <- att_emp_test$riskLevel

str(att_emp_train)
model_1 <- glm(riskLevel ~ ., att_emp_train, family = "binomial")
summary(model_1)

length(model_1$fitted.values)
y_hat_train_class <- ifelse(model_1$fitted.values < 0.5, 0, 1)
length(y_hat_train_class)
length(att_emp_train$riskLevel)
tab_train_class <- table(
  y_hat_train_class,
  att_emp_train$riskLevel,
  dnn = c("Predicted", "Actual")
)

tab_train_class
confusionMatrix(tab_train_class)

y_hat_test_pred_glm <- predict(model_1, test_data_x)
y_hat_test_glm <- as.numeric(y_hat_test_pred_glm > 0.5)
tab_test_class <- table(
  y_hat_test_glm,
  test_data_y,
  dnn = c("Predicted", "Actual")
)

tab_test_class
confusionMatrix(tab_test_class)
## Create dataset for xgboost ##
train_data_x <- subset(att_emp_train, select = -c(riskLevel))
train_data_y <- att_emp_train$riskLevel
train_data_x_matrix <- as.matrix(train_data_x)

# XGboost

model_xg <- xgboost(
  data = train_data_x_matrix,
  label = train_data_y,
  max.depth = 20,
  gamma = 2,
  eta = 1,
  nthread = 2,
  nrounds = 13,
  objective = "binary:logistic"
)

importance_matrix <- xgb.importance(model = model_xg)
importance_matrix
y_hat_xg_train <- predict(model_xg, train_data_x_matrix)
y_hat_xg_train <- as.numeric(y_hat_xg_train > 0.5)
tab_train_class_xg <- table(
  y_hat_xg_train,
  att_emp_train$riskLevel,
  dnn = c("Predicted", "Actual")
)

tab_train_class_xg
confusionMatrix(tab_train_class_xg)

test_data_x_matrix <- as.matrix(test_data_x)

y_hat_xg_test <- predict(model_xg, test_data_x_matrix)
y_hat_xg_test <- as.numeric(y_hat_xg_test > 0.5)
tab_test_class_xg <- table(
  y_hat_xg_test,
  att_emp_test$riskLevel,
  dnn = c("Predicted", "Actual")
)

tab_test_class_xg
confusionMatrix(tab_test_class_xg)
######## XGB Feature selected ##########

new_features <- c(
  "Quarterly.Rating",
  "days_worked",
  "Salary",
  "Age",
  "Total.Business.Value",
  "Joining.Designation",
  "Designation",
  "riskLevel"
)


new_feature_att_emp <- new_att_emp[, names(new_att_emp) %in% new_features]

new_feature_att_emp_y <- new_feature_att_emp$riskLevel
new_feature_att_emp_x <- new_feature_att_emp[, !names(new_feature_att_emp) %in% c("riskLevel")]

new_feature_att_emp_x_mat <- as.matrix(new_feature_att_emp_x)
new_feature_att_emp_y_mat <- as.matrix(new_feature_att_emp_y)
xgb_cv <- xgb.cv(
  data = new_feature_att_emp_x_mat, 
  label = new_feature_att_emp_y, 
  nrounds = 100, 
  nthread = 2, 
  nfold = 10, 
  metrics = list("rmse","auc"),
  max_depth = 20, 
  eta = 1, 
  objective = "binary:logistic",
  prediction = TRUE
)

xgb_cv_pred_y <- xgb_cv$pred
y_hat_xg_cv <- as.numeric(xgb_cv_pred_y > 0.5)
tab_class_xg_cv <- table(
  y_hat_xg_cv,
  new_feature_att_emp_y,
  dnn = c("Predicted", "Actual")
)

confusionMatrix(tab_class_xg_cv)

class0_err <- 1:999
class1_err <- 1:999
overall_err <- 1:999

class0 <- which(new_feature_att_emp_y == 0) 
class1 <- which(new_feature_att_emp_y == 1)
for(i in 1:999) {
  val <- i/1000
  yhat <- ifelse(xgb_cv_pred_y > val, 1, 0)
  overall_err[i] <- mean(new_feature_att_emp_y 
                         != yhat)
  class1_err[i] <- mean(new_feature_att_emp_y[class1] 
                        != yhat[class0])
  class0_err[i] <- mean(new_feature_att_emp_y[class0] 
                        != yhat[class1])
  
  print(class1_err[i], class0_err[i])
}

xrange <- 1:999/1000
class0_err[]
plot(xrange, class0_err, xlab = "Cutoff Value", 
     ylab = "Error Rate", col = "Red", type = "b")
points(xrange, class1_err, xlab = "Cutoff Value", 
       col = "Blue")


xgb_cv_pred_y <- xgb_cv$pred
y_hat_xg_cv <- as.numeric(xgb_cv_pred_y > 0.62)
tab_class_xg_cv <- table(
  y_hat_xg_cv,
  new_feature_att_emp_y,
  dnn = c("Predicted", "Actual")
)

confusionMatrix(tab_class_xg_cv)
###################################################
train.control <- trainControl(method = "LOOCV")
new_feature_att_emp$riskLevel <- as.factor(new_feature_att_emp$riskLevel)
cv_glm <- train(riskLevel ~ ., new_feature_att_emp, method = "glm",
                trControl = train.control)

summary(cv_glm)

tab_class_glm_cv <- table(
  cv_glm$pred$pred,
  new_feature_att_emp$riskLevel,
  dnn = c("Predicted", "Actual")
)
confusionMatrix(tab_class_glm_cv)


