import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
import os
import gc

os.chdir('/Users/gregarbour/Desktop/WiDS Competition/Data Files')
df = pd.read_csv('training_original.csv')
y = df['hospital_death']

df = df.drop(['hospital_death', 'encounter_id', 'patient_id', 'hospital_id', 'icu_id'], axis = 1)
train, test, y_train, y_test = train_test_split(df, y, test_size = 0.2,
                                                       random_state = 42, 
                                                       stratify = y)
y_test.value_counts(normalize = True)
y_train.value_counts(normalize = True)

train = train.reset_index()
test = test.reset_index()

train['is_train'] = 1
test['is_train'] = 0

# Exploration of dataset is only done on Training set. 
# BUT pre-processing like imputation/removing certain variables/
# z-score standardization is performed on the entire dataset
df_all = pd.concat([train, test], axis = 0)
 

#################################
#####     Logic Checks      #####
#################################

#Switch lab values where min > max
for i in range(21, 84):
    error_index = df_all.iloc[:,i*2] < df_all.iloc[:,i*2 + 1]
    error_index = error_index[error_index == True].index
    temp = df_all.iloc[error_index, i*2]
    df_all.iloc[error_index, i*2] = df_all.iloc[error_index, i*2 + 1]
    df_all.iloc[error_index, i*2 + 1] = temp


# See how many negative values in each predictor variable
numeric_vars = df_all.columns[(df_all.dtypes == 'float64') | (df_all.dtypes == 'int64')]
neg_sum = df_all[numeric_vars].apply(lambda x: sum(x < 0), axis = 0)  

# vars that have negative values: apache_4a_hospital_death_prob, apache_4a_icu_death_prob, pre_icu_los_days
train['apache_4a_hospital_death_prob'].hist()
train['apache_4a_icu_death_prob'].hist()
train['pre_icu_los_days'].hist(bins = 30)

#Correct numeric vars with negative values to equal zero
numeric_neg_vars = neg_sum[neg_sum > 0].index
for var in numeric_neg_vars:
    df_all[var][df_all[var]< 0] = 0



#################################
###  Create min/max variables ###
#################################
for i in range(21, 84):
    df_all[re.sub('max', 'min_max', df_all.columns[i*2])] = df_all.iloc[:, i*2] - df_all.iloc[:, i*2 + 1]
    
#Export files to be used for exploratory plots in R
train.to_csv('exploratory_df.csv')
y_train.to_csv('y_train.csv')


    
#################################
##### Bin certain variables #####
#################################

train['apache_2_diagnosis'].value_counts(normalize = True) #Need to look up documentation for this var
train['apache_3j_diagnosis'].round().nunique() # Rounding reduces from 397 to 113 unique values

df_all['apache_3j_diagnosis'].nunique()
df_all['apache_3j_diagnosis'].round().nunique() # Rounding reduces from 399 to 113 unique values for df_all
df_all['apache_3j_diagnosis'] = df_all['apache_3j_diagnosis'].round()


#df_all.to_csv('df_all_pt1.csv')



####################################################
### Remove highly skewed or correlated variables ###
####################################################
# See 'Cleaning pt2.R' file for script to determine which variables to drop
drop_vars = pd.read_csv('drop_vars.csv')
drop_vars = pd.Series(drop_vars['var_name'].unique())
df_all = df_all.drop(drop_vars, axis = 1)
train = df_all.loc[df_all.is_train == 1]


#################################
##### Impute Missing Values #####
#################################

### Exploration of Missing Values ###
num_missing = train.isnull().sum()
num_missing = pd.DataFrame({'var_name': num_missing.index, 
                            'num_missing': num_missing, 
                            'percent_missing': num_missing/len(train.index)})
plt.hist(num_missing['percent_missing'])

# #Can also possibly remove ROWS that have a certain proportion of missingness
row_missing = train.apply(lambda x: sum(pd.isnull(x)), axis = 1)

# 60,000 rows have missingness less than 0.5
plt.hist(row_missing/len(train.columns), bins = 20, cumulative = True)
plt.xlabel('Percentage of Missingness')
plt.title('Row Missingness')
# #But this is a bad idea! Missing values are very likely MNAR so the missingness means something!



### Categorical Variables ###
cat_vars = pd.Series(['apache_2_bodysystem', 'apache_2_diagnosis', 'apache_3j_bodysystem', 'apache_3j_diagnosis', 'ethnicity', 
 'hospital_admit_source', 'icu_admit_source', 'icu_stay_type', 'icu_type'])


# How many null values in the categorical variables?
cat_na_sum = train[cat_vars].isnull().sum()

# Every variable EXCEPT icu_stay_type and icu_type
df_all[cat_vars].isnull().sum() # Same vars as in df_all and train

# Keep those that have at least one missing value (since we're going to do mode imputation next)
cat_vars_w_null_values = pd.Series(cat_na_sum[cat_na_sum > 0].index)

# Mode imputation for all categorical vars (train AND test sets)
# Using mode from train set only
for var in cat_vars_w_null_values:
    mode = train[var].mode()
    na_index = df_all[var].isnull()
    df_all[var].value_counts(dropna = False)
    df_all[var][na_index] = pd.Series(mode).repeat(sum(na_index))
    df_all[var].value_counts(dropna = False)


### Binary Variables ###
bin_vars = pd.Series(['aids', 'apache_post_operative', 'arf_apache', 'cirrhosis', 'diabetes_mellitus', 'elective_surgery', 
                      'gcs_unable_apache', 'male', 'hepatic_failure', 'immunosuppression', 'intubated_apache', 'leukemia', 
                      'solid_tumor_with_metastasis', 'ventilated_apache'])
overlap = set(drop_vars).intersection(set(bin_vars))

# Vector of binary variables AFTER removing those in drop_vars
bin_vars = pd.Series(['apache_post_operative', 'diabetes_mellitus', 'elective_surgery', 
                      'male', 'intubated_apache','ventilated_apache'])
                     
# Binarize gender in train and df_all
train['gender'] = np.where(train['gender'] == 'M', 1, 0)
train = train.rename(columns={"gender": "male"})

df_all['gender'] = np.where(df_all['gender'] == 'M', 1, 0)
df_all = df_all.rename(columns={"gender": "male"})

# How many null values in the Binary variables?
bin_na_sum = pd.DataFrame(columns = ['na_count'], data = train[bin_vars].isnull().sum())
bin_na_sum = bin_na_sum.reindex(columns = ['na_count', 'zero', 'one'])

for i in range(0, bin_na_sum.shape[0]):
    bin_na_sum.iloc[i,1] = train[bin_vars[i]].value_counts(dropna = True)[train[bin_vars[i]].value_counts(dropna = True).index==0].values
    bin_na_sum.iloc[i,2] = train[bin_vars[i]].value_counts(dropna = True)[train[bin_vars[i]].value_counts(dropna = True).index==1].values

# Keep those that have at least one missing value (since we're going to do ??? imputation next)
bin_vars_w_null_values = pd.Series(bin_na_sum[bin_na_sum.na_count > 0].index)



### Explore Remaining variables ###
na_summary = train.isnull().sum()/train.shape[0]

plt.hist(na_summary, bins = 20, cumulative = True, normed = True)
plt.xlabel('Cumulative % of Missingness')
plt.title('Predictor Variable Missingness')

plt.hist(na_summary, bins = 20)
plt.xlabel('% of Missingness')
plt.title('Predictor Variable Missingness')

# Yikes! That's a ton of missing values. All of the variables with missingness > 0.5 are lab values. How these are handled
# will have a huge impact on the model. XG Boost/LGM work with NA's, RF & LR (& NNs?) do not.


### Sets of variables ###
lab_vars = pd.Series([var for var in train.columns if 'h1' in var or 'd1' in var])
apache_vars = pd.Series([var for var in train.columns if 'apache' in var])
lab_apache_vars = pd.concat([lab_vars, apache_vars], axis = 0)
remaining_vars = set(df_all.columns).difference(pd.concat([bin_vars, cat_vars, 
                                                           lab_vars, apache_vars], axis = 0))

# What is the proportion of missing values among these vars?
lab_na_sum = train[lab_vars].isnull().sum()/train.shape[0]
lab_na_sum.hist(normalize = True)
plt.xlabel('Percent Missing')
plt.title("Missing of Lab Values")

apache_na_sum = train[apache_vars].isnull().sum()/train.shape[0]
apache_na_sum.hist(normalize = True)
plt.xlabel('Percent Missing')
plt.title("Missing of APACHE Values")

remaining_na_sum = train[remaining_vars].isnull().sum()/train.shape[0]
remaining_na_sum.hist(normalize = True)
plt.xlabel('Percent Missing')
plt.title("Missing of APACHE Values")

### Impute 0 for missing lab and apache values

from sklearn.impute import SimpleImputer
const_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value = 0)
const_imputer.fit(df_all[lab_apache_vars])
df_all[lab_apache_vars] = const_imputer.transform(df_all[lab_apache_vars])

### KNN imputation of Binary & remaining variables ()
from sklearn.impute import KNNImputer
knn_imputer = KNNImputer(n_neighbors=5)

### One hot encode categorical variables ###
df_all = pd.get_dummies(df_all, dummy_na=True, columns=cat_vars)

# Split out and impute training set first
train = df_all.loc[df_all['is_train']==1]
train.isnull().sum()
train = knn_imputer.fit_transform(train)
train = pd.DataFrame(data = train, columns = df_all.columns)


#Reform df_all as the test set and the imputed training set
test = df_all.loc[df_all['is_train']==0]
df_all = pd.concat([train, test])

# Impute the test set
df_all = knn_imputer.fit_transform(df_all)
df_all = pd.DataFrame(df_all, columns = train.columns)
test = df_all.loc[df_all['is_train'] == 0].drop(['is_train'], axis=1)
train = train.drop(['is_train'], axis = 1)

#Save the results
train.to_pickle('train_knn.pkl')
test.to_pickle('test_knn.pkl')

del df_all
gc.collect()

