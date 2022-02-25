library(ggplot2)
library(dplyr)
library(pastecs)
library(xgboost)
library(lubridate)

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
######### Extract left employees #########
att_emp <- raw_df %>%
            filter(Emp_ID %in% c(att_emp_id))

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



# Remove NA from lag function with 0
att_emp$differenceInSalary[is.na(att_emp$differenceInSalary)] <- 0
att_emp$is_promoted[is.na(att_emp$is_promoted)] <- 0

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

y_hat_test_pred_glm <- predict(model_1, test_data_x)
y_hat_test_glm <- as.numeric(y_hat_test_pred_glm > 0.5)
tab_test_class <- table(
    y_hat_test_glm,
    test_data_y,
    dnn = c("Predicted", "Actual")
)
tab_test_class

## Create dataset for xgboost ##
train_data_x <- subset(att_emp_train, select = -c(riskLevel))
train_data_y <- att_emp_train$riskLevel
train_data_x_matrix <- as.matrix(train_data_x)

# XGboost

model_xg <- xgboost(
    data = train_data_x_matrix,
    label = train_data_y,
    max.depth = 9,
    gamma = 2,
    eta = 1,
    nthread = 2,
    nrounds = 12,
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

test_data_x_matrix <- as.matrix(test_data_x)

y_hat_xg_test <- predict(model_xg, test_data_x_matrix)
y_hat_xg_test <- as.numeric(y_hat_xg_test > 0.5)
tab_test_class_xg <- table(
    y_hat_xg_test,
    att_emp_test$riskLevel,
    dnn = c("Predicted", "Actual")
)
tab_test_class_xg

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
cv <- xgb.cv(data = new_feature_att_emp_x_mat, label = new_feature_att_emp_y, nrounds = 3, nthread = 2, nfold = 5, metrics = list("rmse","auc"),
                  max_depth = 3, eta = 1, objective = "binary:logistic")
