library(ggplot2)
library(readr)
library(dplyr)
library(table1)
library(corrplot)

setwd("~/Desktop/WiDS Competition")

##### Table 1 #####
df <- read_csv('exploratory_df.csv')
df <- select(df, -X1, -index)

y_train <- read_csv('y_train.csv')
df$hospital_death = y_train$hospital_death
remove(y_train)

table1(~ factor(gender) + factor(ethnicity) + age + sodium_apache | hospital_death, data=df)


### Single variable plots
df %>% ggplot(aes(age)) + geom_histogram(bins = n_distinct(df$age) -1) + xlab('Age')
df %>% ggplot(aes(h1_arterial_pco2_min)) + geom_density(fill = 'cyan3') + xlab('Arterial PCO2 Min')

df %>% ggplot(aes(d1_hematocrit_max, fill = factor(hospital_death))) + 
  geom_density(alpha = 0.4) + xlab('Hematocrit Max') +
  labs(fill = "Death in Hospital")

#BMI is very similar between the two levels
df %>% ggplot(aes(bmi, fill = factor(hospital_death))) + 
  geom_density(alpha = 0.4) + xlab('BMI') +
  labs(fill = "Death in Hospital")

# sodium_apache shows more separation and is likely a better variable for prediction the difference
df %>% ggplot(aes(sodium_apache, fill = factor(hospital_death))) + 
  geom_density(alpha = 0.4) + xlab('Sodium Concentration') +
  labs(fill = "Death in Hospital")

# APACHE Probability of Death in Hospital
df %>% filter(apache_4a_hospital_death_prob >= 0) %>% 
  ggplot(aes(apache_4a_hospital_death_prob, fill = factor(hospital_death))) + 
  geom_density(alpha = 0.4) + #xlab('Hematocrit Max') +
  labs(fill = "Death in Hospital")

#### Missingness plots ####
lab_vars <- grep(pattern = "h1|d1", names(df), value = T)
na_sum <- sapply(df, FUN = function(x) sum(is.na(x)))
na_sum <- data.frame(variable = names(na_sum),
                    percent_missing = na_sum/nrow(df))
na_sum$is_lab_var <- factor(ifelse(na_sum$variable %in% lab_vars, 1, 0))

#Overall Missingness
ggplot(na_sum, aes(x = percent_missing)) + geom_histogram() + 
  xlab('% Missingness') + ylab('Number of Variables') + 
  ggtitle('Missingness of All Variables') +
  theme(plot.title = element_text(hjust = 0.5))

#Missingness lab values vs rest of variables
ggplot(na_sum, aes(x = percent_missing, fill = is_lab_var)) + 
  geom_histogram(position = 'stack', bins = 30) +
  ggtitle('Missingness of Lab/Vital Sign vs. Other Variables') +
  theme(plot.title = element_text(hjust = 0.5)) + 
  ylab('Count') + xlab('% Missingness') + labs(fill = 'Lab/Vital Sign')

na_sum2 <- na_sum %>% 
  filter(is_lab_var == 1) %>%
  arrange(percent_missing) %>% 
  mutate(percent_total = seq(from = 0, to = 1, length = n()))

#Explore missingness of lab value variables in particular
na_sum2 %>%  
  ggplot(aes(x = percent_missing, y = percent_total)) + geom_point() +
  xlab('% Missing Values') + ylab('% of Total # of Variables') + 
  ggtitle('Missingness of Lab/Vital Sign Variables') +
  theme(plot.title = element_text(hjust = 0.5))


#Levels of Apache Bodysystem variable
body <- as.data.frame(sort(table(df$apache_3j_bodysystem), decreasing = T))
body %>% ggplot(aes(x = Var1, y = Freq)) + geom_col() +
  xlab('') + ylab('Frequency') + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Effect of imputation on a ;ab value
imputed <- data.frame(d1_hco3_min = df$d1_hco3_min, imputed = 'Original')
imputed <- rbind(imputed,
                 data.frame(d1_hco3_min = ifelse(is.na(df$d1_hco3_min), 0,
                                                          df$d1_hco3_min),
                            imputed = 'Zero Imputed'))

imputed %>% ggplot(aes(x = d1_hco3_min)) + geom_histogram() + facet_wrap(~imputed) +
  xlab('HCO3 Minimum') + ylab('Count')

# Table1 for Aids variable
aids <- data.frame(AIDS = factor(df$aids))
table1(~ AIDS, data = aids)

#### Random Forest Tuning & Evaluation ####
rf <- read_csv("Model Files/All RF Results.csv")

ggplot(rf, aes(x = max_depth, y = AUC)) + geom_point()

ggplot(rf, aes(x = max_features, y = AUC, colour = factor(n_estimators))) + geom_point() +
  labs(color = "Number of Estimators")

ggplot(rf, aes(x = n_estimators, y = AUC, colour = factor(max_depth))) + geom_point() +
  labs(color = "Max Depth") 


# facet: (max_depth), x-axis: max_features, colour: n_estimators
ggplot(rf, aes(x = max_features, y = AUC, colour = factor(n_estimators))) + 
  geom_line() + 
  facet_grid(~max_depth) + 
  xlab("Maximum # of Features") +
  labs(color = "Number of Estimators") +
  ggtitle('AUC of Random Forest Models by Max Depth') +
  theme(plot.title = element_text(hjust = 0.5))


# Showing error bar spread of AUC estimates from each candidate parameter set
rf1 <- rf %>% arrange(AUC) %>% 
  mutate(index = 1:n(),
         y_max = AUC + std_dev,
         y_min = AUC - std_dev)

ggplot(rf1, aes(x = index, y = AUC)) + geom_point() +
  geom_errorbar(data = rf1, mapping = aes(ymin = y_min, ymax = y_max)) +
  xlab('Model Number')

rf2 <- rf1 %>% filter(max_features == 100, n_estimators == 50, index != 57)

ggplot(rf2, aes(x = max_depth, y = AUC)) + geom_point() +
  geom_errorbar(data = rf2, mapping = aes(ymin = y_min, ymax = y_max)) +
  xlab('Max Depth') +
  ggtitle('Standard Errors of AUC by Max Depth (with fixed Max Features & Number of Estimators)') +
  theme(plot.title = element_text(hjust = 0.5))

#Candidate models using the 'simplest model within 1 std error of the best' approach
rf1 %>% filter(AUC > 0.8808738, index != 57) %>% 
  ggplot(aes(x = max_features, y = AUC, colour = factor(n_estimators))) + 
  geom_point() + 
  facet_grid(~max_depth) +
  labs(color = "Number of Estimators") + xlab('Number of Estimators') +
  ggtitle('AUC of Random Forest Models by Max Depth') +
  theme(plot.title = element_text(hjust = 0.5))

# Feature Importance
rf_imp <- read_csv("Desktop/WiDS Competition/Data Files/RF Importance.csv")
xg_imp <- read_csv("Desktop/WiDS Competition/Data Files/XG Importance.csv")
lr_imp <- read_csv("Desktop/WiDS Competition/Data Files/LR Importance.csv")

hist(lr_imp$coefs)


