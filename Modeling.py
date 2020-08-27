import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel

from sklearn.preprocessing import scale
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import log_loss, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance

from joblib import dump, load
import os
import time

os.chdir('/Users/gregarbour/Desktop/WiDS Competition/Data Files')
df = pd.read_csv('training_original.csv')
y = df['hospital_death']

df = df.drop(['hospital_death', 'encounter_id', 'patient_id', 'hospital_id', 'icu_id'], axis = 1)
train, test, y_train, y_test = train_test_split(df, y, test_size = 0.2,
                                                       random_state = 42, 
                                                       stratify = y)

del([df, train, test])

train = pd.read_pickle('train_knn.pkl')
test = pd.read_pickle('test_knn.pkl')

train = train.drop(['hospital_admit_source_Observation',
                    'hospital_admit_source_Other'], axis = 1)
test = test.drop(['hospital_admit_source_Observation',
                    'hospital_admit_source_Other'], axis = 1)
var_names = train.columns

# Standardize train and test sets
train = scale(train)
test = scale(test)

#Change directory as most future objects will be saved here
os.chdir('/Users/gregarbour/Desktop/WiDS Competition/Model Files')



##########################
### Create the k-folds ###
##########################
n_splits = 10
n_repeats = 1
kf = RepeatedKFold(
    n_splits=n_splits, 
    n_repeats=n_repeats, 
    random_state=32)



######################################################################
#### Random Forest with exhaustive grid search & Cross validation ####
######################################################################

### First model ###
# Specify the 1st parameter grid to be used
param_grid_rf = {'n_estimators':[50, 100, 150],
                 'max_depth': [2, 4],
                 'max_features': [20, 50, 100]}

rf_grid = GridSearchCV(RandomForestClassifier(), 
                         param_grid_rf, 
                         cv=kf, 
                         scoring = 'roc_auc',
                         n_jobs = 4)

#Fit the model(s)
start_time = time.time()
rf_fit = rf_grid.fit(train, y_train)
end_time = time.time()
print("Elapsed time: ", end_time - start_time)

#View results
rf_fit.cv_results_['mean_test_score']
rf_fit.cv_results_['params']

# Save results for later use
dump(rf_fit, 'rf_fit.pkl')

# Load results
# rf_fit = load('rf_fit.pkl')

# summarize results
print("Best: %f using %s" % (rf_fit.best_score_, rf_fit.best_params_))
means_rf = rf_fit.cv_results_[ 'mean_test_score' ]
stds_rf = rf_fit.cv_results_[ 'std_test_score' ]
params_rf = rf_fit.cv_results_[ 'params' ]

# Create dataframe of CV results for plotting in R
params_rf = pd.DataFrame.from_dict(params_rf)
params_rf['AUC'] = means_rf
params_rf['std_dev'] = stds_rf
params_rf.to_csv('RF1 results.csv')



### 2nd model ###
# Expand the grid slightly to a different set of parameters in the same neighborhood as best solution
param_grid_rf2 = {'n_estimators':[8,10,12],
                 'max_depth': [8, 10, 12],
                 'max_features': [50, 100, 150]}

rf_grid2 = GridSearchCV(RandomForestClassifier(), 
                         param_grid_rf2, 
                         cv=kf, 
                         scoring = 'roc_auc',
                         n_jobs = 4)

#Fit the model(s)
start_time = time.time()
rf_fit2 = rf_grid2.fit(train, y_train)
end_time = time.time()
print("Elapsed time: ", end_time - start_time)

#View results
rf_fit2.cv_results_['mean_test_score']
rf_fit2.cv_results_['params']

# Save results for later use
dump(rf_fit2, 'rf_fit2.pkl')

# Load results
# rf_fit2 = load('rf_fit2.pkl')

# summarize results
print("Best: %f using %s" % (rf_fit2.best_score_, rf_fit2.best_params_))
means_rf2 = rf_fit2.cv_results_[ 'mean_test_score' ]
stds_rf2 = rf_fit2.cv_results_[ 'std_test_score' ]
params_rf2 = rf_fit2.cv_results_[ 'params' ]

# Create dataframe of CV results for plotting in R
params_rf2 = pd.DataFrame.from_dict(params_rf2)
params_rf2['AUC'] = means_rf2
params_rf2['std_dev'] = stds_rf2
params_rf2.to_csv('RF2 results.csv')



### 3rd model ###
# Expand the grid slightly to a different set of parameters in the same neighborhood as best solution
param_grid_rf3 = {'n_estimators':[50],
                 'max_depth': [8, 10],
                 'max_features': [100]}

rf_grid3 = GridSearchCV(RandomForestClassifier(), 
                         param_grid_rf3, 
                         cv=kf, 
                         scoring = 'roc_auc',
                         n_jobs = 4)


#Fit the model(s)
start_time = time.time()
rf_fit3 = rf_grid3.fit(train, y_train)
end_time = time.time()
print("Elapsed time: ", end_time - start_time)

#View results
rf_fit3.cv_results_['mean_test_score']
rf_fit3.cv_results_['params']

# Save results for later use
dump(rf_fit3, 'rf_fit3.pkl')

# Load results
# rf_fit3 = load('rf_fit3.pkl')

# summarize results
print("Best: %f using %s" % (rf_fit3.best_score_, rf_fit3.best_params_))
means_rf3 = rf_fit3.cv_results_[ 'mean_test_score' ]
stds_rf3 = rf_fit3.cv_results_[ 'std_test_score' ]
params_rf3 = rf_fit3.cv_results_[ 'params' ]

# Create dataframe of CV results for plotting in R
params_rf3 = pd.DataFrame.from_dict(params_rf3)
params_rf3['AUC'] = means_rf3
params_rf3['std_dev'] = stds_rf3
# params_rf3.to_csv('RF3 results.csv')



### Fourth model ###
# Specify the parameter grid to be used
param_grid_rf4 = {'n_estimators':[50, 100, 150],
                 'max_depth': [6, 10, 14],
                 'max_features': [100, 150]}

rf_grid4 = GridSearchCV(RandomForestClassifier(), 
                         param_grid_rf4, 
                         cv=kf, 
                         scoring = 'roc_auc',
                         n_jobs = 4)

#Fit the model(s)
start_time = time.time()
rf_fit4 = rf_grid4.fit(train, y_train)
end_time = time.time()
print("Elapsed time: ", end_time - start_time)

#View results
rf_fit4.cv_results_['mean_test_score']
rf_fit4.cv_results_['params']

# Save results for later use
os.chdir('/Users/gregarbour/Desktop/WiDS Competition/Model Files')
dump(rf_fit4, 'rf_fit4.pkl')

# Load results
# rf_fit4 = load('rf_fit4.pkl')

# summarize results
print("Best: %f using %s" % (rf_fit4.best_score_, rf_fit4.best_params_))
means_rf4 = rf_fit4.cv_results_[ 'mean_test_score' ]
stds_rf4 = rf_fit4.cv_results_[ 'std_test_score' ]
params_rf4 = rf_fit4.cv_results_[ 'params' ]

# Create dataframe of CV results for plotting in R
params_rf4 = pd.DataFrame.from_dict(params_rf4)
params_rf4['AUC'] = means_rf4
params_rf4['std_dev'] = stds_rf4
# params_rf4.to_csv('RF4 results.csv')



### Export CSV of summary of all models
params_rf_all = pd.concat([params_rf, params_rf2, params_rf3, params_rf4])
params_rf_all.to_csv('All RF Results.csv')



### Final Model ###
# Refit model based on best parameters from grid search
rf_final = RandomForestClassifier(max_depth = 10,
                                  max_features = 50,
                                  n_estimators = 50,
                                  n_jobs = 4)
rf_final.fit(train, y_train)

#Save final model for later use
dump(rf_final, 'rf_final.pkl')

# Load results
# rf_final = load('rf_final.pkl')


#Predict on Train Set
y_pred_prob_rf = rf_final.predict_proba(train)[:,1]
roc_auc_score(y_true = y_train, y_score = y_pred_prob_rf)

y_pred_class_rf = rf_final.predict(train)
print(confusion_matrix(y_train, y_pred_class_rf))
print(classification_report(y_train, y_pred_class_rf))

#Feature importance
rf_importance = pd.DataFrame(rf_final.feature_importances_)
rf_importance['variable'] = var_names
# Export to R to plot
rf_importance.to_csv('/Users/gregarbour/Desktop/WiDS Competition/Data Files/RF Importance.csv')



#########################################
#### SVM with exhaustive grid search ####
#########################################
# Parameter grid to search
param_grid_svm = {'C': [0.01, 0.1], 'kernel': ['linear']} 

# Define the base model & Grid search method
svc_grid = GridSearchCV(SVC(), 
                    param_grid_svm, 
                    cv=kf, 
                    scoring = 'roc_auc',
                    n_jobs = 4)

# Fit the models
start_time = time.time()
svc_fit = svc_grid.fit(train, y_train)
end_time = time.time()
print("Elapsed time: ", end_time - start_time)

#View results
svc_fit.cv_results_['mean_test_score']
svc_fit.cv_results_['params']

# Save results for later use
dump(svc_fit, 'svc_fit.pkl')

# Load results
# svc_fit = load('svc_fit.pkl')

# summarize results
print("Best: %f using %s" % (svc_fit.best_score_, svc_fit.best_params_))
means_svc = svc_fit.cv_results_[ 'mean_test_score' ]
stds_svc = svc_fit.cv_results_[ 'std_test_score' ]
params_svc = svc_fit.cv_results_[ 'params' ]



### Final Model ###
# Refit model based on best parameters from grid search
svc_final = SVC(C = 0.01, kernel = 'linear', probability = True)
svc_final.fit(train, y_train)

# Save results for later use
dump(svc_final, 'svc_final.pkl')

# Load results
# svc_final = load('svc_final.pkl')



##################################
####       XG Boost CV        ####
##################################

# Define model and first parameter grid
xg = XGBClassifier()

param_grid_xg = {'max_depth': [4, 8, 12], 
    'learning_rate': [0.01, 0.05],
    'objective': ['binary:logistic'],
    'subsample': [0.3, 0.5],
    'colsample_bytree': [0.4, 0.6, 0.8],
    'gamma': [0],
    'eval_metric': ['auc']}

#Fit the model using randomized grid search
xg_grid = GridSearchCV(xg, param_grid_xg, scoring="roc_auc", n_jobs=-1, cv=kf)
start_time = time.time()
xg_fit = xg_grid.fit(train, y_train)
end_time = time.time()
print("Elapsed time: ", end_time - start_time)

#View results
xg_fit.cv_results_['mean_test_score']
xg_fit.cv_results_['params']

# Save results for later use
dump(xg_fit, 'xg_fit.pkl')

# Load results
# xg_fit = load('xg_fit.pkl')

# summarize results
print("Best: %f using %s" % (xg_fit.best_score_, xg_fit.best_params_))
means_xg = xg_fit.cv_results_[ 'mean_test_score' ]
stds_xg = xg_fit.cv_results_[ 'std_test_score' ]
params_xg = xg_fit.cv_results_[ 'params' ]



### Final XGBoost Model ###
# Refit model based on best parameters from grid search
xg_final = XGBClassifier(colsample_bytree = 0.6, 
                         eval_metric = 'auc', 
                         gamma = 0, 
                         learning_rate = 0.05, 
                         max_depth = 12, 
                         objective = 'binary:logistic',
                         subsample = 0.5)

xg_final.fit(train, y_train)


# Save Final Model for later use
dump(xg_final, 'xg_final.pkl')

# Load results
# xg_final = load('xg_final.pkl')



#Predict on Train Set
y_pred_prob_xg = xg_final.predict_proba(train)[:,1]
roc_auc_score(y_true = y_train, y_score = y_pred_prob_xg)

y_pred_class_xg = rf_final.predict(train)
print(confusion_matrix(y_train, y_pred_class_xg))
print(classification_report(y_train, y_pred_class_xg))


#Feature importance
xg_importance = pd.DataFrame(xg_final.feature_importances_)
xg_importance['variable'] = var_names
# Export to R to plot
xg_importance.to_csv('/Users/gregarbour/Desktop/WiDS Competition/Data Files/XG Importance.csv')


######################################
#### Elastic Logistic Regression  ####
######################################
lr= SGDClassifier(loss = 'log', penalty = 'elasticnet')

param_grid_lr = {'alpha': [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2], 
                 'l1_ratio': [0.2, 0.4, 0.6, 0.8]}

param_grid_lr = {'alpha': [0.1], 
                 'l1_ratio': [0.4]}
lr_grid = GridSearchCV(lr, 
                       param_grid_lr, 
                       cv=kf, 
                       scoring = 'roc_auc',
                       n_jobs = 4)
lr_fit = lr_grid.fit(train, y_train)


#View results
lr_fit.cv_results_['mean_test_score']
lr_fit.cv_results_['params']

# Save results for later use
dump(lr_fit, 'lr_fit.pkl')

# Load results
# lr_fit = load('lr_fit.pkl')

# Summarize results
print("Best: %f using %s" % (lr_fit.best_score_, lr_fit.best_params_))
means_lr = lr_fit.cv_results_[ 'mean_test_score' ]
stds_lr = lr_fit.cv_results_[ 'std_test_score' ]
params_lr = lr_fit.cv_results_[ 'params' ]
params_index = np.arange(1, 1 + len(params_lr))


### Final LR Model ###
lr_final = SGDClassifier(loss = 'log', 
                         penalty = 'elasticnet',
                         alpha = 0.1,
                         l1_ratio = 0.4)
lr_final = lr_final.fit(train, y_train)

# Save results for later use
dump(lr_final, 'lr_final.pkl')

# Load results
lr_final = load('lr_final.pkl')


#Feature importance
lr_importance = pd.DataFrame(lr_final.coef_.T, columns = ['coefs'])
lr_importance['variable'] = var_names
# Export to R to plot
lr_importance.to_csv('/Users/gregarbour/Desktop/WiDS Competition/Data Files/LR Importance.csv')



##################################
####  Predict w Final Models  ####
##################################

# Predict Random Forest on Test Set
rf_prob = rf_final.predict_proba(test)[:,1]
roc_auc_score(y_true = y_test, y_score = rf_prob)

rf_class = rf_final.predict(test)
print(confusion_matrix(y_test, rf_class))
print(classification_report(y_test, rf_class))

# Predict XGBoost on Test Set
xg_prob = xg_final.predict_proba(test)[:,1]
roc_auc_score(y_true = y_test, y_score = xg_prob)

xg_class = xg_final.predict(test)
print(confusion_matrix(y_test, xg_class))

# Predict SVC on Test Set
svc_prob = svc_final.predict_proba(test)[:,1]
roc_auc_score(y_true = y_test, y_score = svc_prob)

svc_class = svc_final.predict(test)
print(confusion_matrix(y_test, svc_class))

# Predict LR on Test Set
lr_prob = lr_final.predict_proba(test)[:,1]
roc_auc_score(y_true = y_test, y_score = lr_prob)

lr_class = lr_final.predict(test)
print(confusion_matrix(y_test, lr_class))
