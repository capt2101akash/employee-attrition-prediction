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
setwd('/Users/garvit/Documents/GitHub/employee-attrition-prediction')
raw_df  = read.csv("dataset/train_data.csv")
#att_emp_id = raw_df[raw_df$LastWorkingDate != "", 2]
att_emp = raw_df
#att_emp <- raw_df %>%
#  filter(Emp_ID %in% c(att_emp_id))

# Feature Engineering
att_emp <- att_emp %>%
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

att_emp$differenceInSalary[is.na(att_emp$differenceInSalary)] <- 0
att_emp$is_promoted[is.na(att_emp$is_promoted)] <- 0

# Create Factors for qualitative variable
category_cars <- c("Gender", "Education_Level", "is_promoted",'LastWorkingDate')
att_emp$Gender <- as.factor(att_emp$Gender)
att_emp$Education_Level <- as.factor(att_emp$Education_Level)
att_emp$is_promoted <- as.factor(att_emp$is_promoted)
att_emp$LastWorkingDate<-as.Date(att_emp$LastWorkingDate)
empid = att_emp$Emp_ID
str(att_emp)
drop_cols <- c(
  "MMM.YY","Dateofjoining","City","Emp_ID"
)
new_att_emp <- att_emp[, !names(att_emp) %in% drop_cols]
gender_dummy <- model.matrix(~Gender - 1, data = new_att_emp)
education_dummy <- model.matrix(~Education_Level - 1, data = new_att_emp)
is_promoted_dummy <- model.matrix(~is_promoted - 1, data = new_att_emp)

new_att_emp <- cbind(empid,new_att_emp, gender_dummy,education_dummy,is_promoted_dummy)

new_att_emp <- new_att_emp[, !names(new_att_emp) %in% c("Gender", "Education_Level", "is_promoted")]
head(new_att_emp)
str(new_att_emp)
drop2 <-c('is_promoted0','Education_LevelMaster','GenderMale')
new_att_emp <- new_att_emp[, !names(new_att_emp) %in% drop2]
str(new_att_emp)
new_att_emp$LastWorkingDate <-as.character(new_att_emp$LastWorkingDate)
new_att_emp$LastWorkingDate[is.na(new_att_emp$LastWorkingDate)] <- 0
new_att_emp$LastWorkingDate[!new_att_emp$LastWorkingDate=='0']<-1
new_att_emp$LastWorkingDate = as.numeric(new_att_emp$LastWorkingDate)
new_att_emp$DaysBeforeDeparture[is.na(new_att_emp$DaysBeforeDeparture)] <- 0

try1 = new_att_emp %>%
  group_by(empid)
try2 = try1 %>% summarize(
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
try2

set.seed(33)
index = createDataPartition(try2$LWD,p=0.7,list=FALSE)
train_data = try2[index,]
train_data = train_data[,-1]
test_data = try2[-index,]
test_data = test_data[,-1]
table(try2$LWD)

#Logistic Regression without oversampling
log_base_train = glm(LWD~.,train_data,family='binomial')
summary(log_base_train)
length(log_base_train$fitted.values)
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


#Logistic regression with oversampling 
smote = SMOTE(train_data[,-13],train_data$LWD)
oversampled_train = smote$data
names(oversampled_train)[13] = "LWD"
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

#Feature Selection 
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

#KNN clustering 



#odata = oversample(try2,method = "SMOTE",classAttr = 'LWD')
odata = SMOTE(try2[,-14] ,try2[,14])
#odata = ovun.sample(LWD~., data = try2,method='SMOTE',p=0.5,N=3232)
glmbase = glm(LWD~.-empid,try2,family='binomial')
sum = summary(glmbase)
exp(coef(glmbase))
glmbase2 = glm(LWD~ JoiningDesignation+ QuaterlyRating + Ispromoted ,try2,family='binomial')
summary(glmbase2)
#variable selection with forward selection and backward 
#knn clustering
table(odata$LWD)

