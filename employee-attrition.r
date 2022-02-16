library(ggplot2)
library(dplyr)
library(pastecs)

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


att_emp$differenceInSalary[is.na(att_emp$differenceInSalary)] <- 0

category_cars <- c("Gender", "Education_Level")

att_emp$Gender <- as.factor(att_emp$Gender)
att_emp$Education_Level <- as.factor(att_emp$Education_Level)

str(att_emp)

drop_cols <- c(
    "MMM.YYY",
    "Emp_ID",
    "City",
    "DateofJoining",
    "LastWorkingDate",
    "Joining.Destination"
)

new_att_emp <- att_emp[, !names(att_emp) %in% drop_cols]

str(new_att_emp)

new_att_emp <- new_att_emp %>%
    mutate(
        riskLevel = if_else(
            DaysBeforeDeparture < 180,
            1,
            0
        )
    )

