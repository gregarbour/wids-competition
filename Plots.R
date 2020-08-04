library(ggplot2)
library(readr)
library(dplyr)

setwd("~/Desktop/Survey Paper")

#### Random Forest Tuning & Evaluation
params_rf1 <- read_csv("RF1 results.csv")
params_rf2 <- read_csv("RF2 results.csv")

# facet: (max_depth), x-axis: max_features, colour: n_estimators
ggplot(params_rf1, aes(x = max_features, y = AUC, colour = factor(n_estimators))) + 
  geom_line() + 
  facet_grid(~max_depth) + 
  xlab("Maximum # of Features") +
  labs(color = "Number of Estimators")

# 2nd parameter set CV results
ggplot(params_rf2, aes(x = max_features, y = AUC, colour = factor(n_estimators))) + 
  geom_line() + 
  facet_grid(~max_depth) + 
  xlab("Maximum # of Features") +
  labs(color = "Number of Estimators")

# Showing error bar spread of AUC estimates from each candidate parameter set
params_rf1 <- params_rf1 %>% arrange(AUC) %>% 
  mutate(index = 1:n(),
         y_max = AUC + std_dev,
         y_min = AUC - std_dev)

ggplot(params_rf1, aes(x = index, y = AUC)) + geom_point() +
  geom_errorbar(data = params_rf1, mapping = aes(ymin = y_min, ymax = y_max))
