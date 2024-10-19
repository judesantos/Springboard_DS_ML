import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from bayes_opt import BayesianOptimization
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.base import BaseEstimator, ClassifierMixin

verbose = 1
random_state = 42

criterion_str = lambda n: 'gini' if n == 0 else ('entropy' if n == 1 else 'log_loss')

# Enhanced data splitting which makes sure duplicates do not get copied into the test dataset
#
def train_test_split_with_duplicates(data, target: str, test_size=0.25, random_state=None):
    """Train/test split data while ensuring no duplicates are introduced in the test set"""
    # Identify duplicates
    duplicate_indices = data.duplicated(keep=False)
    # Separate data into duplicates and non-duplicates
    data_no_duplicates = data[~duplicate_indices]
    data_duplicates = data[duplicate_indices]
    # Split the data without duplicates first
    X_train, x_test, y_train, y_test = train_test_split(
        data_no_duplicates.drop(columns=[target], axis=1),
        data_no_duplicates[target],
        test_size=test_size,
        random_state=random_state
    )
    # Add duplicates to the training set
    if not data_duplicates.empty:
        X_train = pd.concat([X_train, data_duplicates.drop(columns=[target])], axis=0)
        y_train = pd.concat([y_train, data_duplicates[target]], axis=0)
    
    return X_train, x_test, y_train, y_test

# Function for Random Forest Optimization
#
def rf_eval(n_estimators, max_depth, min_samples_split, criterion, X, y, x_t, y_t):
    """Run Evaluation on Random Forest Model with input hyperparameters"""
    params = {
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        'min_samples_split': int(min_samples_split),
        'criterion': criterion_str(criterion),
        'random_state': random_state,
        'verbose': 0,
        'n_jobs': -1
    }
    model = RandomForestClassifier(**params)
    model.fit(X, y)
    y_pred = model.predict(x_t)
    accuracy = accuracy_score(y_t, y_pred)
    
    return accuracy

# Setup Optimization for Random Forest
#
def rf_optimizer(X, y, x_t, y_t):
    """Random Forest Model Hyperparams optimizer""" 
    # Define hyperparameter boundaries
    rf_param_grid = {
        'n_estimators': (100, 500),
        'max_depth': (3, 150),
        'min_samples_split': (2, 50),
        'criterion': (0, 1),  # 0 for 'gini', 1 for 'entropy', 'log_loss'
    }
    # Initialize optimizer
    opt = BayesianOptimization(
        f=lambda n_estimators, max_depth, min_samples_split, criterion: rf_eval(n_estimators, max_depth, min_samples_split, criterion, X, y, x_t, y_t),
        pbounds=rf_param_grid,
        random_state=random_state,
        verbose=verbose,
    )
    # Run
    opt.maximize(init_points=5, n_iter=15)
    
    return opt

# Evaluate performance of RFC model
#
def train_rf_model(rf_opt, X, y, x_t, y_t):
    """Train and score best RF model"""
    best_rf_params = rf_opt.max['params']
    criterion = criterion_str(best_rf_params['criterion'])
    best_rf_model = RandomForestClassifier(
        n_estimators=int(best_rf_params['n_estimators']),
        max_depth=int(best_rf_params['max_depth']),
        min_samples_split=int(best_rf_params['min_samples_split']),
        criterion=criterion,
        random_state=random_state,
        verbose=0,
        n_jobs=-1
    )
    # Train the best model
    best_rf_model.fit(X, y)
    # Evaluate the best model
    final_rf_predictions = best_rf_model.predict(x_t)
    final_rf_accuracy = accuracy_score(y_t, final_rf_predictions)
    print("Accuracy: ", final_rf_accuracy)

    return final_rf_predictions, best_rf_model

# Optimize XGB model
#
def xgb_eval(n_estimators, max_depth, learning_rate, gamma, X, y, x_t, y_t):
    """Run Evaluation on XGB Model with input hyperparameters"""
    params = {
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        'learning_rate': learning_rate,
        'gamma': gamma,
        'objective': 'binary:logistic',
        'random_state': random_state,
        'eval_metric': 'logloss',
        'n_jobs': -1
    }
    # Train the model
    model = XGBClassifier(**params)
    model.fit(X, y)
    # Predict and evaluate
    y_pred = model.predict(x_t)
    accuracy = accuracy_score(y_t, y_pred)
    
    return accuracy
	
# Optimization for XGB
#
def xgb_optimizer(X, y, x_t, y_t):
    """XGB Model Hyperparams optimizer""" 
    # Define hyperparameter boundaries
    xgb_param_grid = {
        'n_estimators': (50, 200),
        'max_depth': (3, 12),
        'learning_rate': (0.01, 0.3),
        'gamma': (0, 5)
    }
    # Initialize optimizer
    opt = BayesianOptimization(
        f=lambda n_estimators, max_depth, learning_rate, gamma: xgb_eval(n_estimators, max_depth, learning_rate, gamma, X, y, x_t, y_t),
        pbounds=xgb_param_grid,
        random_state=random_state,
        verbose=verbose
    )
    # Run
    opt.maximize(init_points=5, n_iter=15)

    return opt

def train_xgb_model(xgb_opt, X, y, x_t, y_t):
    """Train and score best XGB model"""
    best_xgb_params = xgb_opt.max['params']
    best_xgb_model = XGBClassifier(
        objective='binary:logistic',
        n_estimators=int(best_xgb_params['n_estimators']),
        max_depth=int(best_xgb_params['max_depth']),
        learning_rate=best_xgb_params['learning_rate'],
        gamma=best_xgb_params['gamma'],
        random_state=random_state,
        n_jobs=-1
    )
    # Train the best model
    best_xgb_model.fit(X, y)
    # Evaluate the best model
    final_xgb_predictions = best_xgb_model.predict(x_t)
    final_xgb_accuracy = accuracy_score(y_t, final_xgb_predictions)
    print("Accuracy: ", final_xgb_accuracy)
    
    return final_xgb_predictions, best_xgb_model

# Create evaluator to optimize our LGBM model
#

objective_str = lambda n: 'binary' if 0 == n else ('multiclass' if 1 == n else 'regression')
metric_str = lambda n: 'auc' if 0 == n else 'binary_logloss'
boosting_type_str = lambda n: 'gbdt' if 0 == n else ('dart' if n == 1 else ('goss' if n == 2 else ('rf' if n == 3 else 'gbdt')))
is_unbalance_b = lambda b: False if b == 0 else True

#def lgb_eval(num_leaves, max_depth, learning_rate, n_estimators, X, y, x_t, y_t):
def lgb_eval(objective, metric, is_unbalance, num_leaves, max_depth, learning_rate, n_estimators, boosting_type, lambda_l2, lambda_l1, min_child_samples, min_data_in_leaf, X, y, x_t, y_t):
    """Run Evaluation on LGBM Model with input hyperparameters"""
    params = {
        'objective': objective_str(objective),
        'metric': metric_str(metric),
        'is_unbalance': is_unbalance_b(is_unbalance),
        'num_leaves': int(num_leaves),
        'max_depth': int(max_depth),
        'lambda_l2': float(lambda_l2),
        'lambda_l1': float(lambda_l1),
        'min_child_samples': int(min_child_samples),
        'min_data_in_leaf': int(min_data_in_leaf),
        'learning_rate': float(learning_rate),
        'n_estimators': int(n_estimators),
        'bagging_fraction': 0.1, #1.0,
        'bagging_freq': 10, #10),  # Must be an integer
        'boosting_type': boosting_type_str(boosting_type),
        'random_state': random_state,
        'subsample_freq': 5,
        'bagging_seed': 42,
        'bagging_freq': 5,
        'verbosity': -1,
        'vebose':-1,
        'num_threads': 20,
        'n_jobs': -1
    }
    #params = {
    #    'objective': 'binary',
    #    'metric': 'binary_logloss',
    #    'num_leaves': int(num_leaves),
    #    'max_depth': int(max_depth),
    #    'learning_rate': learning_rate,
    #    'n_estimators': int(n_estimators),
    #    'boosting_type': 'gbdt',
    #    'random_state': random_state,
    #    'verbose': -1,
    #    'n_jobs': -1
    #}
    # Train the model
    model = LGBMClassifier(**params)
    model.fit(X, y)
    # Predict and evaluate
    y_pred = model.predict(x_t)
    accuracy = accuracy_score(y_t, y_pred)
    
    return accuracy

# Setup Optimization for LGBM
#
def lgb_optimizer(X, y, x_t, y_t):
    """LGBM Model Hyperparams optimizer""" 
    lgb_param_grid = {
        'objective': (0, 0), #('binary', 'binary')
        'metric': (0, 1), #('auc', 'binary_logloss'(
        'is_unbalance': (0, 1), # True, False
        'num_leaves': (25, 4000),
        'max_depth': (5, 63),
        'lambda_l2': (0.0, 10),
        'lambda_l1': (0.0, 10),
        'min_child_samples': (50, 10000),
        'min_data_in_leaf': (100, 2000),
        'learning_rate': (0.01, 0.1),
        'n_estimators': (50, 200),
        'boosting_type': (0, 1) # ('gbdt', 'goss')
    }
    #lgb_param_grid = {
    #    'num_leaves': (20, 150),
    #    'max_depth': (3, 12),
    #    'learning_rate': (0.01, 0.1),
    #    'n_estimators': (50, 200)
    #}
    opt = BayesianOptimization(
        f=lambda objective, metric, is_unbalance, num_leaves, max_depth, learning_rate, n_estimators, boosting_type, lambda_l2,lambda_l1, min_child_samples, min_data_in_leaf: lgb_eval(objective, metric, is_unbalance, num_leaves, max_depth, learning_rate, n_estimators, boosting_type, lambda_l2,lambda_l1, min_child_samples, min_data_in_leaf, X, y, x_t, y_t),
        pbounds=lgb_param_grid,
        random_state=random_state,
        verbose=0
    )
    opt.maximize(init_points=5, n_iter=15)

    return opt

def train_lgb_model(lgb_opt, X, y, x_t, y_t):
    """Train and score best LGBM model"""
    best_lgb_params = lgb_opt.max['params']
    best_lgb_model = LGBMClassifier(
        objective=objective_str(best_lgb_params['objective']),
        metric=metric_str(best_lgb_params['metric']),
        is_unbalance=is_unbalance_b(best_lgb_params['is_unbalance']),
        num_leaves=int(best_lgb_params['num_leaves']),
        max_depth=int(best_lgb_params['max_depth']),
        learning_rate=float(best_lgb_params['learning_rate']),
        n_estimators=int(best_lgb_params['n_estimators']),
        lambda_l2=float(best_lgb_params['lambda_l2']),
        lambda_l1=float(best_lgb_params['lambda_l1']),
        min_child_samples=int(best_lgb_params['min_child_samples']),
        min_data_in_leaf=int(best_lgb_params['min_data_in_leaf']),
        boosting_type=boosting_type_str(best_lgb_params['boosting_type']),
        random_state=random_state,
        verbose=-1,
        n_jobs=-1
    )
    # Train the best model
    best_lgb_model.fit(X, y)
    # Evaluate the best model
    final_lgb_predictions = best_lgb_model.predict(x_t)
    final_lgb_accuracy = accuracy_score(y_t, final_lgb_predictions)
    print(f'Accuracy: {final_lgb_accuracy}')

    return final_lgb_predictions, best_lgb_model

def stacked_models(stacked_models, X, y):
    """Stack Models using StackingClassifier and LogisticRegression as meta-learner"""
    # Create the stacking model with a logistic regression meta-learner
    stacking_model = StackingClassifier(
        estimators=stacked_models,
        final_estimator=LogisticRegression(random_state=random_state),
        verbose=0
    )
    stacking_model.fit(X, y)

    return stacking_model
	
# Print confusion, classification metrics
#
def print_model_metrics(y_t, y_p):
    cm = confusion_matrix(y_t, y_p)
    print(f'Confusion Matrix:\n{cm}')
    report = classification_report(y_t, y_p)
    print(f'\nClassification Report:\n{report}')

    return cm

# Define stack model optimizer, evaluator
#
def train_stacked_models(df, target):
    """
    Create and optimize models in RFC, XGB, LGBM, stack the 3 models and get a final StackClassifier:
    - Run Bayesian optimizer for each classifier using pre-selected hyper parameters
    - Run each classifier with optimized parameter values taken from previous step. Print score.
    - Stack the best model of each classifier into a StackClassifier using LogisticRegression as meta-learner
    - Train the stack model.
    - Print metrics
    - Return the stack model. Performance should be enhanced.
    """
    # Train/Test Split
    X_tr, x_ts, y_tr, y_ts = train_test_split_with_duplicates(df, target) 
    print('1. Hyperparameter optimization')
    print('==============================')
    # Find optimal params for LGBM
    print('LGBM:')
    lgb_opt = lgb_optimizer(X_tr, y_tr, x_ts, y_ts)
    best_lgb_pred, best_lgb_model = train_lgb_model(lgb_opt, X_tr, y_tr, x_ts, y_ts)
    # Find optimal params for XGB
    print('XGB:')
    xgb_opt = xgb_optimizer(X_tr, y_tr, x_ts, y_ts)
    best_xgb_pred, best_xgb_model = train_xgb_model(xgb_opt, X_tr, y_tr, x_ts, y_ts)
    print('Random Forest:')
    rf_opt = rf_optimizer(X_tr, y_tr, x_ts, y_ts)
    best_rf_pred, best_rf_model = train_rf_model(rf_opt, X_tr, y_tr, x_ts, y_ts)
    print('\n2. Model Performance')
    print('====================')
    print('LGB:')
    print_model_metrics(y_ts, best_lgb_pred)
    print('XGB:')
    print_model_metrics(y_ts, best_xgb_pred)
    print('Random Forest:')
    print_model_metrics(y_ts, best_rf_pred)
    print('\n3. Stack Model')
    print('==============')
    stck_model = stacked_models(
        [
            ('rf', best_rf_model),
            ('xgb', best_xgb_model),
            ('lgb', best_lgb_model),
        ],
        X_tr,
        y_tr
    )
    # Evaluate the stacking model
    stck_pred = stck_model.predict(x_ts)
    stck_accuracy = accuracy_score(y_ts, stck_pred)
    #print("Accuracy: ", stck_accuracy)
    print_model_metrics(y_ts, stck_pred)
    
    return stck_model
    
class WeightedStackingClassifier(BaseEstimator, ClassifierMixin):
        """
        Weighted stacking classifier - combines a tuple of base estimators
        to test the combined predictive performance by allowing a flexible
        weighting of each base estimator predictions.
        Parameters
        ----------
        estimators : list of tuples
            A tuple is composed of the name (str) and an estimator instance.
        final_estimator : estimator object
            The estimator that trains on the weighted predictions of the base estimators.
        weights : list of float, optional
            List of weights of each base estimator.
            Each weight defines the contribution of each base estimator predictions 
            to the final estimator prediction. 
            If None, assign equal weights.
        """
        def __init__(self, estimators, final_estimator, weights=None):
            self.estimators = estimators
            self.final_estimator = final_estimator
            self.weights = weights if weights is not None else [1] * len(estimators)
            
        def fit(self, X, y):
            # Fit each base estimator
            self.base_estimators_ = []
            for _, estimator in self.estimators:
                estimator.fit(X, y)
                self.base_estimators_.append(estimator)
            # Collect all base predictions
            self.predictions_ = np.array([estimator.predict(X) for estimator in self.base_estimators_])
            # Weighted mean predictions
            weighted_predictions = np.tensordot(self.weights, self.predictions_, axes=(0, 0))
            self.final_estimator.fit(weighted_predictions.reshape(-1, 1), y)
            return self
            
        def predict(self, X):
            # Call predict for each base estimator
            base_preds = np.array([estimator.predict(X) for estimator in self.base_estimators_])
            # Get weighted mean predictions
            weighted_preds = np.tensordot(self.weights, base_preds, axes=(0, 0))
            return self.final_estimator.predict(weighted_preds.reshape(-1, 1))

