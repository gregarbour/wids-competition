library(tidyverse)
library(corrplot)
library(caret)

df <- read_csv("~/Desktop/Survey Paper/Code/Data Files/df_all_pt1.csv")
df <- df %>% filter(is_train == 1) %>%  select(-X1, -index, -is_train)


#### Which variables are extremely skewed towards only one value
zero_vars <- caret::nearZeroVar(df)
zero_names <- names(df)[zero_vars]
#Remove the 35 suggested vars
df <- select(df, -all_of(zero_names))

##### Which variables are highly correlated and can be removed?
numeric_vars <- read_csv("~/Desktop/Survey Paper/Code/Data Files/numeric_vars.csv",
                        col_names = 'variable_name')
numeric_vars <- c(numeric_vars$variable_name, 
                  grep(names(df), pattern = "_min_max", value = T))

df2 <- select(df, all_of(numeric_vars))

correlations <- cor(df2, use = 'pairwise.complete.obs')
correlations <- as.data.frame(correlations)
corr_90p <- sapply(correlations, FUN = function(x)sum(x > 0.9))

#Using algorithm outlines in Kuhn book, these are the predictors recommended for elimination
highCorr <- findCorrelation(correlations, cutoff = 0.9) 
length(highCorr)/ncol(df) #28% of the columns!!!!

#Try a higher cutoff value
highCorr2 <- findCorrelation(correlations, cutoff = 0.95)
length(highCorr2)/ncol(df2) #Still 25% 

#Try an even higher cutoff value
highCorr3 <- findCorrelation(correlations, cutoff = 0.975)
length(highCorr3)/ncol(df2) #18% !!!! Got a lot of correlated shit here, jeez
names(df2)[highCorr3]

correlated_vars = names(df2)[highCorr2]
drop_vars = bind_rows(data.frame(var_name = zero_names, type = 'zeroVar'),
                      data.frame(var_name = correlated_vars, 
                                 type = 'correlatedVar'))

write.csv(drop_vars, file = '~/Desktop/Survey Paper/Code/Data Files/drop_vars.csv')





