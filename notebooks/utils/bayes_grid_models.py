#param drid models for BayessearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

#skopt
lr_param_grid = {
  'lr__C': Real(1e-3, 100, 'log-uniform'),
  'lr__fit_intercept': Categorical([True, False]), 
  'lr__max_iter': Integer(1e+2, 1e+5, 'log-uniform'),
  'lr__penalty':  Categorical(['l1', 'l2']),
  'lr__solver': Categorical(['liblinear', 'saga']),
  'lr__tol': Real(1e-5, 1e-3, 'log-uniform'),
  'lr__class_weight':  Categorical([None, 'balanced']),

}




#skopt
xgb_param_grid = {
        'xgb__colsample_bylevel' : Real(1e-1, 1, 'uniform'),
        'xgb__colsample_bytree' : Real(6e-1, 1, 'uniform'),
        'xgb__gamma' :  Real(5e-1, 6, 'log-uniform'),
        'xgb__learning_rate' : Real(10**-5, 10**0, "log-uniform"),
        'xgb__max_depth' : Integer(1, 25, 'uniform'),
        'xgb__min_child_weight' : Integer(1, 10, 'uniform'),
        'xgb__n_estimators' : Integer(50, 400, 'log-uniform'),
        'xgb__reg_alpha' : Real(1e-2, 1, 'log-uniform'),
        'xgb__reg_lambda' : Real(1e-2, 1, 'log-uniform'),
        'xgb__subsample' : Real(6e-1, 1, 'uniform'),
        'xgb__scale_pos_weight' : Integer(1, 10000, 'log-uniform')
        
}



#skopt
lgbm_param_grid = {
    'lgbm__boosting_type': Categorical(['gbdt', 'goss', 'dart']),
    'lgbm__num_leaves':  Integer(2, 10, 'uniform'),
    'lgbm__learning_rate': Real(10**-5, 10**0, "log-uniform"),
    'lgbm__subsample_for_bin': Integer(20000, 300000, 'log-uniform'),
    'lgbm__min_child_samples': Integer(20, 500, 'uniform'),
    'lgbm__reg_alpha': Real(10**-3, 10**0, "log-uniform"),
    'lgbm__reg_lambda': Real(10**-3, 10**0, "log-uniform"),
    'lgbm__colsample_bytree': Real(6e-1, 1, 'uniform'),
    'lgbm__subsample': Real(5e-1, 1, 'uniform'),
    'lgbm__is_unbalance': Categorical([True, False]),
    #'lgbm__class_weight': Categorical([None, 'balanced'])

}



#############################################################################
#############################################################################
#############################################################################
#############################################################################



