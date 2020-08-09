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

# Standardize train and test sets
train = scale(train)
test = scale(test)

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
param_grid_rf = {'n_estimators':[8,10,12],
                 'max_depth': [2, 4, 6, 8],
                 'max_features': [15, 20, 30, 50, 100, 150]}

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
os.chdir('/Users/gregarbour/Desktop/WiDS Competition/Model Files')
dump(rf_fit, 'rf_fit.pkl')

# Load results
# rf_fit = load('rf_fit.pkl')

# summarize results
print("Best: %f using %s" % (rf_fit.best_score_, rf_fit.best_params_))
means_rf = rf_fit.cv_results_[ 'mean_test_score' ]
stds_rf = rf_fit.cv_results_[ 'std_test_score' ]
params_rf = rf_fit.cv_results_[ 'params' ]

# Create dataframe of CV results for plotting in R
# params_rf = pd.DataFrame.from_dict(params_rf)
# params_rf['AUC'] = means_rf
# params_rf['std_dev'] = stds_rf
# params_rf.to_csv('RF1 results.csv')



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
rf_fit2 = load('rf_fit2.pkl')

# summarize results
print("Best: %f using %s" % (rf_fit2.best_score_, rf_fit2.best_params_))
means_rf2 = rf_fit2.cv_results_[ 'mean_test_score' ]
stds_rf2 = rf_fit2.cv_results_[ 'std_test_score' ]
params_rf2 = rf_fit2.cv_results_[ 'params' ]

# Create dataframe of CV results for plotting in R
# params_rf2 = pd.DataFrame.from_dict(params_rf2)
# params_rf2['AUC'] = means_rf2
# params_rf2['std_dev'] = stds_rf2
# params_rf2.to_csv('RF2 results.csv')


### Final Model ###
# Refit model based on best parameters from grid search
rf_final = RandomForestClassifier(max_depth = 8,
                                  max_features = 100,
                                  n_estimators = 12,
                                  n_jobs = 4)
rf_final.fit(train, y_train)

#Predict on Train Set
y_pred_prob_rf = rf_final.predict_proba(train)[:,1]
roc_auc_score(y_true = y_train, y_score = y_pred_prob_rf)

y_pred_class_rf = rf_final.predict(train)
print(confusion_matrix(y_train, y_pred_class_rf))
print(classification_report(y_train, y_pred_class_rf))




########################
#### SVM Classifier ####
########################
# svm_model = SVC(gamma='auto', probability = True)
# clf_svm = make_pipeline(StandardScaler(), svm_model)
# clf_svm.fit(train, y_train)

# y_pred_svm = clf_svm.predict_proba(train)[:,1]
# roc_auc_score(y_true = y_train, y_score = y_pred_svm)


#########################################
#### SVM with exhaustive grid search ####
#########################################
# Parameter grid to search
param_grid_svm = {'C': [0.1, 1], 'kernel': ['linear']}

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

# Create dataframe of CV results for plotting in R
# params_svc = pd.DataFrame.from_dict(params_svc)
# params_svc['AUC'] = means_svc
# params_svc['std_dev'] = stds_svc
# params_svc.to_csv('SVC results.csv')

### Final Model ###
# Refit model based on best parameters from grid search
svc_final = RandomForestClassifier(max_depth = 8,
                                  max_features = 100,
                                  n_estimators = 12,
                                  n_jobs = 4)
svc_final.fit(train, y_train)

#Predict on Train Set
y_pred_prob_svc = svc_final.predict_proba(train)[:,1]
roc_auc_score(y_true = y_train, y_score = y_pred_prob_svc)

y_pred_class_svc = svc_final.predict(train)
print(confusion_matrix(y_train, y_pred_class_svc))
print(classification_report(y_train, y_pred_class_svc))



##################################
####      Base XG Boost       ####
##################################

xg = XGBClassifier(nthread = -1, 
                   random_state = 0)
eval_set = [(train, y_train)]
xg.fit(train, y_train, verbose = True, eval_metric = 'auc', eval_set = eval_set)
# Next step = to get verbose to work, need to also specify an eval set.
# I.e. need k-fold CV and eval on each hold out fold



# feature importance
xg.booster().get_score(importance_type='gain') ### Doesn't work! Fix!

# plot feature importance
plot_importance(xg, max_num_features = 20)
pyplot.show()

thresholds = xg.feature_importances_
thresholds.sort()
thresholds = thresholds[thresholds > 0]
for thresh in thresholds:
  # select features using threshold
  selection = SelectFromModel(xg, threshold=thresh, prefit=True)
  select_X_train = selection.transform(train)
  
  # train model
  selection_model = XGBClassifier()
  selection_model.fit(select_X_train, y_train)
  
  # eval model
  select_X_test = selection.transform(train)
  predictions = selection_model.predict(select_X_test)
  accuracy = accuracy_score(y_train, predictions)
  print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1],
      accuracy*100.0))

# Compare results from 1st and 2nd models
probs1 = xg.predict_proba(train)[:,1]
roc_auc_score(y_true = y_train, y_score = probs1)



##################################
####       XG Boost CV        ####
##################################

# Define model and parameter grid
xg = XGBClassifier()

param_grid_xg = {'max_depth': [4, 6, 8, 10], 
    'learning_rate': [0.01, 0.05, 0.1],
    'objective': 'binary:logistic',
    'subsample': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'colsample_bytree': [0.4, 0.6, 0.8, 1],
    'gamma': [0],
    'eval_metric': 'auc'}

#Fit the model using randomized grid search
xg_grid = GridSearchCV(xg, param_grid_xg, scoring="roc_auc", n_jobs=-1, cv=kf)
xg_fit = xg_grid.fit(train, y_train)

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

# Create dataframe of CV results for plotting in R
# params_xg = pd.DataFrame.from_dict(params_xg)
# params_xg['AUC'] = means_xg
# params_xg['std_dev'] = stds_xg
# params_xg.to_csv('XG results.csv')

### Final Model ###
# Refit model based on best parameters from grid search
xg_final = XGBClassifier()
xg_final.fit(train, y_train)

#Predict on Train Set
y_pred_prob_xg = xg_final.predict_proba(train)[:,1]
roc_auc_score(y_true = y_train, y_score = y_pred_prob_xg)

y_pred_class_xg = rf_final.predict(train)
print(confusion_matrix(y_train, y_pred_class_xg))
print(classification_report(y_train, y_pred_class_xg))


######################################
#### Elastic Logistic Regression  ####
######################################
lr= SGDClassifier(loss = 'log', penalty = 'elasticnet')

param_grid_lr = {'alpha': [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 0.1], 
                 'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}

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



