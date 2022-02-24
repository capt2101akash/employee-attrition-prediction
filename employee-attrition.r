library(ggplot2)
library(dplyr)
library(pastecs)
library(xgboost)

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

ggplot(gender_count, aes(x = Gender, y = count)) +
geom_bar(stat = "identity", fill = "Blue")

ggplot(city_count, aes(x = City, y = count)) +
geom_bar(stat = "identity", fill = "green")


att_emp_id <- raw_df[raw_df$LastWorkingDate != "", 2]

att_emp <- raw_df %>%
            filter(Emp_ID %in% c(att_emp_id))


raw_df %>%
    mutate(
        label = if_else(Emp_ID %in% c(att_emp_id), 1, 0),
        color = if_else(Emp_ID %in% c(att_emp_id), "red", "yellow")
    ) %>%
    ggplot(aes(x = Age, y = Salary, colour = color)) +
    geom_point()



att_emp <- att_emp %>%
    group_by(Emp_ID) %>%
    mutate(
        DaysBeforeDeparture = as.integer(as.Date(max(LastWorkingDate)) - as.Date(MMM.YY))
    ) %>%
    mutate(
        differenceInSalary = Salary - lag(Salary)
    )

att_emp[, 2:15] %>%
    filter(Emp_ID == 4)

att_emp$differenceInSalary[is.na(att_emp$differenceInSalary)] <- 0

category_cars <- c("Gender", "Education_Level")

att_emp$Gender <- as.factor(att_emp$Gender)
att_emp$Education_Level <- as.factor(att_emp$Education_Level)

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


new_att_emp <- new_att_emp %>%
    mutate(
        riskLevel = if_else(
            DaysBeforeDeparture < 120,
            1,
            0
        )
    )

gender_dummy <- model.matrix(~Gender - 1, data = new_att_emp)
education_dummy <- model.matrix(~Education_Level - 1, data = new_att_emp)

new_att_emp <- cbind(new_att_emp, gender_dummy)
new_att_emp <- cbind(new_att_emp, education_dummy)
new_att_emp <- new_att_emp[, !names(new_att_emp) %in% c("Gender", "Education_Level")]

new_att_emp$riskLevel = as.integer(new_att_emp$riskLevel)
str(new_att_emp)
cor(new_att_emp)[9, ]

head(new_att_emp)

new_att_emp %>%
ggplot(aes(x = riskLevel)) +
geom_bar()

new_att_emp %>%
ggplot(aes(x = DaysBeforeDeparture, y = Total.Business.Value)) +
geom_point(aes(color = riskLevel)) +
labs(title = "Relation between Business Value v/s Departure Date")

table(new_att_emp$riskLevel)

new_att_emp %>%
ggplot(aes(x = DaysBeforeDeparture, y = Total.Business.Value)) +
geom_point(aes(color = riskLevel)) +
labs(title = "Relation between Business Value v/s Departure Date")

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

new_att_emp %>%
    ggplot(aes(x = Salary)) +
    geom_histogram()

train_sample <- sample(nrow(new_att_emp), 0.8*nrow(new_att_emp))

att_emp_train <- new_att_emp[train_sample, ]

att_emp_test <- new_att_emp[-train_sample, ]
table(att_emp_train$riskLevel)
table(att_emp_test$riskLevel)

# check base model
base_model <- glm(riskLevel ~ ., att_emp_train, family = "binomial")

summary(base_model)

att_emp_train <- att_emp_train[, !names(att_emp_train) %in% c("GenderMale", "Education_LevelMaster", "DaysBeforeDeparture")]

att_emp_test <- att_emp_test[,
                !names(att_emp_test) %in% c(
                "GenderMale", "Education_LevelMaster", "DaysBeforeDeparture"
                )
            ]

test_data_x <- subset(att_emp_test, select = -c(riskLevel))
test_data_y <- att_emp_test$riskLevel

model_1 <- glm(riskLevel ~ ., att_emp_train, family = "binomial")
summary(model_1)


y_hat_train_class <- ifelse(model_1$fitted.values < 0.5, 0, 1)
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
train_data_x <- subset(att_emp_train, select = -c(riskLevel))
train_data_y <- att_emp_train$riskLevel
train_data_x_matrix <- as.matrix(train_data_x)

# XGboost

model_xg <- xgboost(
    data = train_data_x_matrix,
    label = train_data_y,
    max.depth = 9,
    eta = 1,
    nthread = 2,
    nrounds = 30,
    objective = "binary:logistic"
)

importance_matrix <- xgb.importance(model = model_xg)
importance_matrix
y_hat_xg_train <- predict(model_xg, train_data_x_matrix)
y_hat_xg_train <- as.numeric(y_hat_xg_train > 0.5)
y_hat_xg_train
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

summary(model_xg)
