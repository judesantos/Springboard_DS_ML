import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import shap

from scipy.stats import chi2_contingency
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif  # For classificationo

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import make_scorer, roc_auc_score, f1_score, log_loss

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from bayes_opt import BayesianOptimization
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.base import BaseEstimator, ClassifierMixin 


from tpot import TPOTClassifier

verbose = 1
random_state = 42
n_iter = 15
test_size = 0.3

criterion_str = lambda n: 'gini' if n == 0 else ('entropy' if n == 1 else 'log_loss')
solver_str = lambda n: 'liblinear' if n == 0 else ('lbfgs' if n == 1 else 'sag')

def train_test_split_with_duplicates(data, target: str, test_size=test_size, random_state=None):
    """Train/test split data while ensuring no duplicates are introduced in the test set"""
    # Identify duplicates
    duplicate_indices = data.duplicated(keep=False)
    # Separate data into duplicates and non-duplicates
    data_no_duplicates = data[~duplicate_indices]
    data_duplicates = data[duplicate_indices]
    #print(f'#Duplicates: {len(data_duplicates)}')
    #print(f'#Non-duplicates: {len(data_no_duplicates)}')
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

    X = pd.concat([X_train, x_test])
    y = pd.concat([y_train, y_test])

    return X_train, x_test, y_train, y_test, X, y

def lr_eval(C, solver, X, y, x_t, y_t):
    """Run evaluation on LogisticRegression with input hyperparameters"""
    model = LogisticRegression(
        class_weight='balanced',
        C=C,
        solver=solver_str(solver), 
        random_state=random_state,
        max_iter=n_iter*150,
        verbose=0,
        n_jobs=-1
    )
    model.fit(X, y)
    y_pred = model.predict(x_t)
    accuracy = accuracy_score(y_t, y_pred)
    
    return accuracy

def lr_optimizer(X, y, x_t, y_t):
    """LogisiticRegression Model Hyperparameter optimizer"""
    # Define h-param bounds
    lr_param_grid = {
        'C': (0.001, 10),  # Regularization parameter
        'solver': (1, 2)   # Use integer encoding for solvers
    }
    # Initialize optimizer
    opt = BayesianOptimization(
        f=lambda C, solver: lr_eval(C, solver, X, y, x_t, y_t),
        pbounds=lr_param_grid,
        random_state=random_state,
        verbose=verbose,
    )
    # Run
    opt.maximize(init_points=5, n_iter=1)
    
    return opt

def train_lr_model(lr_opt, X, y, x_t, y_t):
    """Train and score best RF model"""
    best_lr_params = lr_opt.max['params']
    best_lr_model = LogisticRegression(
        class_weight='balanced',
        C=best_lr_params['C'],
        solver=solver_str(best_lr_params['solver']),
        random_state=random_state,
        verbose=0,
        n_jobs=-1
    )
    # Train the best model
    best_lr_model.fit(X, y)
    # Evaluate the best model
    final_lr_predictions = best_lr_model.predict(x_t)
    #final_lr_accuracy = accuracy_score(y_t, final_lr_predictions)
    #print('Accuracy: ', final_lr_accuracy)

    return final_lr_predictions, best_lr_model

def rf_eval(n_estimators, max_depth, min_samples_split, criterion, X, y, x_t, y_t):
    """Run Evaluation on Random Forest Model with input hyperparameters"""
    params = {
        'class_weight': 'balanced',
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
        f=lambda n_estimators, max_depth, min_samples_split, criterion: 
            rf_eval(n_estimators, max_depth, min_samples_split, criterion, X, y, x_t, y_t),
        pbounds=rf_param_grid,
        random_state=random_state,
        verbose=verbose,
    )
    # Run
    opt.maximize(init_points=5, n_iter=n_iter)
    
    return opt

def train_rf_model(rf_opt, X, y, x_t, y_t):
    """Train and score best RF model"""
    best_rf_params = rf_opt.max['params']
    criterion = criterion_str(best_rf_params['criterion'])
    best_rf_model = RandomForestClassifier(
        class_weight='balanced',
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
    #final_rf_accuracy = accuracy_score(y_t, final_rf_predictions)
    #print('Accuracy: ', final_rf_accuracy)

    return final_rf_predictions, best_rf_model

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
        f=lambda n_estimators, max_depth, learning_rate, gamma: 
            xgb_eval(n_estimators, max_depth, learning_rate, gamma, X, y, x_t, y_t),
        pbounds=xgb_param_grid,
        random_state=random_state,
        verbose=verbose
    )
    # Run
    opt.maximize(init_points=5, n_iter=n_iter)

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
    #final_xgb_accuracy = accuracy_score(y_t, final_xgb_predictions)
    #print('Accuracy: ', final_xgb_accuracy)
    
    return final_xgb_predictions, best_xgb_model

# Create evaluator to optimize our LGBM model
#

objective_str = lambda n: 'binary' if 0 == n else ('multiclass' if 1 == n else 'regression')
metric_str = lambda n: 'auc' if 0 == n else 'binary_logloss'
boosting_type_str = lambda n: 'gbdt' if 0 == n else ('dart' if n == 1 
                        else ('goss' if n == 2 else ('rf' if n == 3 else 'gbdt')))
is_unbalance_b = lambda b: False if b == 0 else True

#def lgb_eval(num_leaves, max_depth, learning_rate, n_estimators, X, y, x_t, y_t):
def lgb_eval(objective, metric, is_unbalance, num_leaves, max_depth, learning_rate, 
        n_estimators, boosting_type, lambda_l2, lambda_l1, min_child_samples, 
            min_data_in_leaf, X, y, x_t, y_t):
    """
    Run Evaluation on LGBM Model with input hyperparameters
    """
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
    opt = BayesianOptimization(
        f=lambda objective, metric, is_unbalance, num_leaves, max_depth, learning_rate, 
            n_estimators, boosting_type, lambda_l2,lambda_l1, min_child_samples, min_data_in_leaf: 
                lgb_eval(objective, metric, is_unbalance, num_leaves, max_depth, learning_rate,
                    n_estimators, boosting_type, lambda_l2,lambda_l1, min_child_samples, 
                        min_data_in_leaf, X, y, x_t, y_t),
        pbounds=lgb_param_grid,
        random_state=random_state,
        verbose=0
    )
    opt.maximize(init_points=5, n_iter=n_iter)

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
    #final_lgb_accuracy = accuracy_score(y_t, final_lgb_predictions)
    #print(f'Accuracy: {final_lgb_accuracy}')

    return final_lgb_predictions, best_lgb_model

def stacked_models(stacked_models, X, y):
    """Stack Models using StackingClassifier and LogisticRegression as meta-learner"""
    # Create the stacking model with a logistic regression meta-learner
    stacking_model = StackingClassifier(
        estimators=stacked_models,
        final_estimator=LogisticRegression(
            random_state=random_state,
            n_jobs=-1
        ),
        verbose=0
    )
    stacking_model.fit(X, y)

    return stacking_model
	
def print_model_metrics(y_t, y_p):
    """Print out metrics"""
    cm = confusion_matrix(y_t, y_p)
    print(f'Confusion Matrix:\n{cm}')
    report = classification_report(y_t, y_p)
    print(f'\nClassification Report:\n{report}')

    return cm

def train_model(mh_obj, cb_optimizer, cb_trainer) -> tuple[object, object, float]:
    """
    Create and optimize model:
    - Run Bayesian optimizer for each classifier using pre-selected hyper parameters
    - Run given classifier with optimized parameter values taken from previous step. Print score.
    - Print metrics

    returns:
    =======
    tuple | (y_prediction:object, model:object, accuracy:float)
    """
    # Get splits
    X_tr, x_ts, y_tr, y_ts = mh_obj.X_train, mh_obj.x_test, mh_obj.y_train, mh_obj.y_test
    
    # Find optimal params for LR
    optimized = cb_optimizer(X_tr, y_tr, x_ts, y_ts)
    best_pred, best_model = cb_trainer(optimized, X_tr, y_tr, x_ts, y_ts)
    accuracy = accuracy_score(y_ts, best_pred)

    return best_pred, best_model, accuracy

def train_binary_classifiers(mh_obj):
    """
    Create and optimize models in RFC, XGB, LGBM, stack the 3 models and get a final StackClassifier:
    - Run Bayesian optimizer for each classifier using pre-selected hyper parameters
    - Run each classifier with optimized parameter values taken from previous step. Print score.
    - Stack the best model of each classifier into a StackClassifier using LogisticRegression as meta-learner
    - Train the stack model.
    - Print metrics
    """
    # Get splits
    X_tr, x_ts, y_tr, y_ts = mh_obj.X_train, mh_obj.x_test, mh_obj.y_train, mh_obj.y_test
    
    print('1. Hyperparameter optimization')
    print('==============================')
    # Find optimal params for LR
    #print('Logistic Regression:')
    #best_lr_pred, _, accuracy = train_model(mh_obj, lr_optimizer, train_lr_model)
    #print(f'Accuracy: {accuracy}')
    ## Find optimal params for LGBM
    print('LGBM:')
    best_lgb_pred, _, accuracy = train_model(mh_obj, lgb_optimizer, train_lgb_model)
    print(f'Accuracy: {accuracy}')
    # Find optimal params for XGB
    print('XGB:')
    best_xgb_pred, _, accuracy = train_model(mh_obj, xgb_optimizer, train_xgb_model)
    print(f'Accuracy: {accuracy}')
    print('Random Forest:')
    best_rf_pred, _, accuracy = train_model(mh_obj, rf_optimizer, train_rf_model)
    print(f'Accuracy: {accuracy}')
    
    print('\n2. Model Performance')
    print('====================')
    #print('Logistic Regression:')
    #print_model_metrics(y_ts, best_lr_pred)
    print('LGB:')
    print_model_metrics(y_ts, best_lgb_pred)
    print('XGB:')
    print_model_metrics(y_ts, best_xgb_pred)
    print('Random Forest:')
    print_model_metrics(y_ts, best_rf_pred)
    
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

class MentalHealthData:
    """
    Mental Health Data class - handles pre-processing, encoding and splitting of train and test data.
    Feature-engineering is applied only after the split to prevent leakage.
    """
    # Static attr.: Binary dictionary
    yes_no_num = {
        'Yes': 1,
        'No': 0
    }
    
    def __init__(self, mh_df, test_size=test_size, random_state=234):
        
        self.__df = mh_df
        self.__test_size = test_size
        self.random_state = random_state
        self.__new_cols = []
        
        self.X_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.X = None
        self.y = None
    
    def cross_validate_model(self, cb_model_instance) -> object:
        """
        Evaluate a given model returned cb_model_instance

        Returns:
        -------
        object | The cross-val score
        """
        model = cb_model_instance()
        scores = cross_val_score(model, self.X, self.y, cv=5)

        return scores

    def get_data(self):
        """Get transformed data"""
        return self.__df

    def pre_process(self, cb_preprocess):
        """Custom step. Apply external pre-processing step as needed"""  
        self.__new_cols = cb_preprocess(self.__df)

    def train_test_split(self, target='treatment') -> None:
        """
        Train/Test Split Mental health data set, then pre-process train and test data separately
        """
        # First drop 'Timestamp' column, then transform features with pre-defined steps.
        self.__df = self.__df.drop(labels='Timestamp', axis=1)
        # Convert target - a binary categorical feature to numeric
        self.__df[target] = self.__df[target].apply(lambda x: MentalHealthData.yes_no_num.get(x, x))
        # Transform!
        self.__df = self.__fit_transform(self.__df)
        # Split train and test data while making sure that duplicates 
        # only stays in the Train set and not the test set.
        self.X_train, self.x_test, self.y_train, self.y_test, self.X, self.y = train_test_split_with_duplicates(
            self.__df, 
            target,
            test_size=self.__test_size, 
            random_state=self.random_state
        )
    
    def __fit_transform(self, df) -> object:
        """
        Pre-process identified train data features for the mental health data set.
        All identified features were thoroughly analyzed and evaluated prior 
        to making the decision to implement the transformations outlined here.
        Transformation Steps:
        - Filling NaN values in the 'self_employed' column with 'Unknown' value.
        - Convert binary categories('Yes', No') for the features: 
            'Coping_Struggles', 'family_history' to numeric 1 and 0 respectively.
        - Hot-encode the remaining unprocessed features
        """
        bin_cols = []
        _df = df

        # fillna with value 'Unknown' 
        _df['self_employed'] = _df['self_employed'].fillna('Unknown')

        # convert 'Yes', 'No' to 1, 0 - Coping_Struggles, family_history, treatment
        for col in df:
            if len(df[col].unique()) == 2 and col != 'Gender':
                df[col] = df[col].apply(lambda x: MentalHealthData.yes_no_num.get(x, x))

        # one-hot encode
        categorical_cols = df.select_dtypes(include='object').columns
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        df_encoded.columns = df_encoded.columns.str.replace(' ', '_') 

        return df_encoded.astype(int)

def train_report_scores(mh_obj, model: BaseEstimator, target='treatment', plot=False) -> tuple[str, float, float, object]:
    """
    Trains a ML model based on a given dataset and evaluates its performance.
    Print evaluation report
    
    Parameters:
    ----------
    data : pd.DataFrame
        A dataframe containing the features and target data
    target : str
        Name of the column in the data that contains the target labels
    model : BaseEstimator interface
        An instance of the sklearn model to be trained. 
    plot : bool
        Display graphs
    test_size : float, optional, default is 0.3
        Fraction of the data to use as the test dataset.
    random_state : int, optional, default is None
        Controls randomness of the split.
        
    Returns: 
    -------
    tuple : (str, float, float)
    """

    # Get splits
    if isinstance(mh_obj, MentalHealthData):
        X_train, x_test, y_train, y_test = mh_obj.X_train, mh_obj.x_test, mh_obj.y_train, mh_obj.y_test
    else:
        X_train, x_test, y_train, y_test = train_test_split_with_duplicates(
            mh_obj, 
            target 
        )

    # Train, evaluate model
    #

    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict probabilities and classes for test set
    y_probs = model.predict_proba(x_test)[:, 1]  # Probability of the positive class
    y_pred = model.predict(x_test)

    # Check that the target has at least 2 classes, otherwise this is failed.
    vc = y_test.value_counts()
    if vc.count() == 1:
        return (
            f'Test failed! The test variable only has one category ({"Yes" if vc.keys()[0] == 1 else "No"}).',
            0,
            0
        )

    # Calculate F1 Score
    f1 = f1_score(y_test, y_pred)
    # Calculate AUC
    auc = roc_auc_score(y_test, y_probs)
    _score = accuracy_score(y_test, y_pred)
    custom_scorer = make_scorer(f1_score, average='weighted')
    cv_scores = cross_val_score(model, mh_obj.X, mh_obj.y, scoring=custom_scorer, cv=5)
    
    if f1 != 1.0 and auc != 1.0:
        if plot == True:
            # Print Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            print('\nConfusion Matrix (Treatment Required):')
            print(f' No: {cm[0]}')
            print(f'Yes: {cm[1]}')
            
            # Print Classification Report
            report = classification_report(y_test, y_pred)
            print('\nClassification Report:')
            print(report)

            # Plot precision-recall curve
            #
            
            # Calculate precision and recall
            precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
            # Calculate average precision score
            avg_precision = average_precision_score(y_test, y_probs)
            fig, ax = plt.subplots(1, 2, figsize=(6, 3))
            
            ax[0].plot(recall, precision, marker='.', 
                     label='Precision-Recall curve (AP={:.2f})'.format(avg_precision))
            ax[0].set_title('Precision-Recall Curve')
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.grid()
    
            # Plot ROC Curve
            #
            
            RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax[1])
            ax[1].set_title('ROC Curve')
            plt.grid()

            plt.tight_layout()
            _ = plt.show()

            print(f'Cross-V: {cv_scores.mean():.4f}')
    
    return ('', _score, auc, cv_scores)

# Create a callback function for the mental-health object custom step.
# Create feature from 'Timestamp'.
def pre_process_timestamp(df):
  # First, convert 'Timestamp' to datetime
  df['Timestamp'] = pd.to_datetime(df['Timestamp'])

  # Create year, month, day, and hour columns
  df['year'] = df['Timestamp'].dt.year
  df['month'] = df['Timestamp'].dt.month
  df['day'] = df['Timestamp'].dt.day    
  df['hour'] = df['Timestamp'].dt.hour  
  df['minute'] = df['Timestamp'].dt.minute   
  # Convert to ordinal values
  df['year'] = df['year'].astype(int)  
  df['month'] = df['month'].astype(int) 
  df['day'] = df['day'].astype(int)     
  df['hour'] = df['hour'].astype(int)   
  df['minute'] = df['minute'].astype(int)

  return ['year', 'month', 'day', 'hour', 'minute']

def mental_health_instance(df):
  mh_o = MentalHealthData(df) 
  # Apply timestamp feature creation before we drop the timestamp column
  mh_o.pre_process(pre_process_timestamp)
  # Pre-processing the data, then split
  mh_o.train_test_split()

  return mh_o

def evaluate_by_feature_country(df, callback_get_model):
    
    # Get All Country feature names
    #countries = [col for col in df.columns if col.startswith('Country_')]
    countries = df.Country.unique()
    results = [];

    for country in countries:
        country_df = df[df['Country'] == country]
        mho_country = mental_health_instance(country_df.copy()) 
        # Process if we have at least 1 'YES' or 1 'NO', 
        # otherwise we cannot create a model off this country data.
        vc = country_df.treatment.value_counts()
        if vc.count() > 1:
            # Create new model each time 
            model = callback_get_model()
            msg, acc, auc, cv = train_report_scores(mho_country, model) 
            results.append([country, f'{acc:.3f}', f'{auc:.3f}', f'{np.mean(cv):.3f}', f'{np.std(cv):.3f}', msg])
        else:
            message = f'One class found: [{"Yes" if vc.keys()[0] == 1 else "No"}]. '
            message += f'>1 class is required for modeling.'
            results.append([country, '', '', '', '', message])

    columns = ['Country', 'Accuracy', 'AUC', 'CV-F1 (mean)', 'CV-F1 (std)', 'Other']
    df = pd.DataFrame(results, columns=columns)

    return df
    
def TPOT_test(mh_df):
    """
    Tree Based Pipeline Optimization Test
    Run performance, optimization tests on split data.
    """

    X_train, x_test, y_train, y_test = mh_df.X_train, mh_df.x_test, mh_df.y_train, mh_df.y_test

    # Define metrics to evaluate
    metrics = {
        'accuracy': 'accuracy',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'precision': 'precision',
        'recall': 'recall'
    }

    # Dictionary to hold the results
    results = {}

    for metric_name, metric in metrics.items():

        print(f'\nOptimizing for {metric_name}...')
        tpot = TPOTClassifier(verbosity=2, generations=5, population_size=20, random_state=42, scoring=metric)
        tpot.fit(X_train, y_train)
        
        # Evaluate on validation data
        score = tpot.score(x_test, y_test)
        
        # Store results
        results[metric_name] = {
            'best_pipeline': tpot.fitted_pipeline_,
            'score': score
        }
        
        # Get cross-validation score
        cv_scores = cross_val_score(tpot.fitted_pipeline_, X_train, y_train, cv=5)
        results[metric_name]['cv_score'] = cv_scores.mean()
        
        # Make predictions
        predictions = tpot.predict(x_test)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, predictions)
        results[metric_name]['confusion_matrix'] = cm
        
        # Get classification report
        class_report = classification_report(y_test, predictions)
        results[metric_name]['classification_report'] = class_report
        
        print(f'Best {metric_name} Score: ', score)
        print(f'Cross-Validation Score: ', results[metric_name]['cv_score'])
        print('Confusion Matrix:\n', cm)
        print('Classification Report:\n', class_report)

        # Inspect the fitted pipeline
        best_pipeline = tpot.fitted_pipeline_
        print('\nBest Pipeline Steps:')
        
        # If a pipeline with steps
        if hasattr(best_pipeline, 'named_steps'):
            # Iterate through all named steps
            for name, step in best_pipeline.named_steps.items():
                print(f'Step Name: {name}')
                print('Step Type: ', type(step).__name__)
                print('Step Parameters: ', step.get_params())
        else:
            # If it's a single model
            print('Final Classifier Type: ', type(best_pipeline).__name__)
            print('Classifier Parameters: ', best_pipeline.get_params())

    # Get results for each metric
    for metric_name, result in results.items():
        print(f'\n--- Results for {metric_name} ---')
        print('Best Pipeline: ', result['best_pipeline'])
        print('Best Score: ', result['score'])
        print('Cross-Validation Score: ', result['cv_score'])
        print('Confusion Matrix:\n', result['confusion_matrix'])
        print('Classification Report:\n', result['classification_report'])

def analyze_tradeoffs(mh_o, cb_get_model):

    opt_threshold, precision, recall, thresholds, fitted_model = get_optimal_thresholds(
        mh_o,
        cb_get_model,
        metric='f1'
    )
    # Prepare to store TP and FN
    results = []

    pos_th = [(idx, t) for idx, t in enumerate(thresholds) if t >= opt_threshold][:10]
    neg_th = []#[(idx, t) for idx, t in enumerate(thresholds) if t < opt_threshold][:5]
    _thresholds = neg_th + pos_th
    # Reclassify model using the optimized TH.
    for idx, threshold in _thresholds:
        _, _, _, _, cm, _ = reclassify_model(
            mh_o,
            fitted_model,
            threshold
        )
        results.append({
            'Threshold': threshold, 
            'Precision': f'{precision[idx]:.3f}',
            'Recall': f'{recall[idx]:.3f}',
            'True Positives': cm[1, 1], 
            'False Negatives': cm[1, 0]
        })
    
    return pd.DataFrame(results)
    

def get_prc_thresholds(mh_o, cb_get_model):
    """Get precision_recall curve values, including test data predictions and the fitted model"""
     # Get the train classifier
    X_train, x_test, y_train, y_test = mh_o.X_train, mh_o.x_test, mh_o.y_train, mh_o.y_test
    # Get our model and fit data
    model = cb_get_model()
    model.fit(X_train, y_train)
    # Predict 
    y_pred = model.predict_proba(x_test)[:, 1]
    # return precision, recall, threshold 
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

    return model, y_pred, precision, recall, thresholds 


def get_optimal_thresholds(mh_df, cb_get_model, metric='f1'):
    """
    Get optimal precision/recall threshold for a binary category, with option to 
    maximize precision, recall, or balanced precision/recall (f1).
    Parameters:
    ----------
    mh_df | MentalHealthDataset - The split, trained data
    cb_get_model | callback - Get instance of the model to test
    metric | str - Specify threshold focus: 
                1. 'recall' - Maximize True Positives, 
                2. 'precision' - Maximize True Negatives,
                3. 'f1' - Balanced precision
    threshold | float - A user specified threshold value. Overrides criterion.
    Returns:
    -------
    optimal_threshold | float - The optimal threshold value
    precision | list - The list of precision values
    recall | list - The list of recall values
    model | object - The fitted model
    """
    # Get model, prediction and precision/recall values
    model, y_pred, precision, recall, thresholds = get_prc_thresholds(mh_df, cb_get_model)

    scores = []
    if metric == 'f1':
        # Get F1 score for each threshold
        scores = [f1_score(mh_df.y_test, y_pred >= t) for t in thresholds]
    elif metric == 'precision':
        scores = precision
    elif metric == 'recall':
        scores = recall
    
    if len(scores) <= 0:
        return None, None

    # Get the optimal threshold (where F1 is maximum)
    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Return the optimal TH and model
    return optimal_threshold, precision, recall, thresholds, model

def reclassify_model(mh_df, model, threshold: float):
    """
    Reclassify a model based on a class-1 precision/recall threshold. 
    Print results.
    Parameters:
    ----------
    mh_df | MentalHealthData - The trained dataset
    model | object - The fitted model instance
    threshold | float - The optimal precision/recall threshold value
    Returns:
    -------
    y_pred | List of class 1 predictions
    precision | float - The optimal precision value
    recall | float - The optimal recall value
    loss | float - The log loss value
    conf_matrix | List of confusion matrix
    class_report | str - The formatted classification report
    """
    # Get test sets
    x_test, y_test = mh_df.x_test, mh_df.y_test
    y_scores = model.predict_proba(x_test)[:, 1]
    # Reclassify based on the optimal threshold
    y_pred = (y_scores >= threshold).astype(int)
    loss = log_loss(y_test, y_pred)
    opt_precision = precision_score(y_test, y_pred, zero_division=1)
    opt_recall = recall_score(y_test, y_pred)
    #  Evaluate performance
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    # Return results
    return y_pred, opt_precision, opt_recall, loss, conf_matrix, class_report

def print_classification_report(conf_mat, class_rep, precision=None,
    recall=None, x_test=None, model=None, y_test=None, y_pred=None,
        opt_precision=0.0, opt_recall=0.0, log_loss=None, opt_th=None):
    """
    Print, plot the binary classification details for a given model.
    Parameters:
    ----------
    conf_mat | list     - The confusion matrix
    class_rep | object  - The classification report data
    precision | list    - Precision data
    recall | list       - Recall data
    y_test | list       - The target test data
    y_pred | list - The prediction data
    th | float          - If applied, the optimal value of precision and recall.
    """
    print('Confusion Matrix:')
    print(conf_mat)
    print('\nClassification Report:')
    print(class_rep)
    if log_loss != None:
        print(f'\nLog loss: {log_loss:0.3f}')
    if opt_th != None:
        print(f'Optimal Threshold: {opt_th:0.3f}')
    
    fig = None
    ax = []
    if precision.any() and recall.any() and model != None \
            and x_test.empty == False and y_test.empty == False:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    if  precision.any() and recall.any():
        _fig = None
        if fig == None:
            _fig = plt.figure((6, 3))
        elif ax.any():
            _fig = ax[0]

        _fig.plot(recall, precision, marker='.')
        if opt_recall > 0 and opt_precision > 0:
            _fig.plot(opt_recall, opt_precision, 'ro', markersize=8, label='Optimal TH')
            _fig.legend()

        if ax.any():
            _fig.set_title('Precision-Recall Curve')
            _fig.set_xlabel('Recall')
            _fig.set_ylabel('Precision')
        else:
            _fig.title('Precision-Recall Curve')
            _fig.xlabel('Recall')
            _fig.ylabel('Precision')
        _fig.grid() 
    
    if model != None and x_test.empty == False and y_test.empty == False:
        _fig_roc = None
        if fig == None: 
            _fig = plt.figure(6, 3)
        else:
            _fig_roc = ax[1]

        RocCurveDisplay.from_estimator(model, x_test, y_test, ax=_fig_roc)
        _fig_roc.set_title('ROC Curve')
        _fig_roc.grid()

    plt.tight_layout()
    _ = plt.show()

def evaluate_precision_recall_optimized_model(mh_o, cb_model, metric='f1'):
    """
    - Get optimal precision/recall threshold for a binary category, with option to 
        maximize precision, recall, or balanced precision/recall (f1).
    - Reclassify the fitted model using the optimal thresholds
    - Print classification report results
    """
    # Train model and optimize for recall. 
    # Return optimized threshold, including the fitted model 
    opt_threshold, precision, recall, _, fitted_model = get_optimal_thresholds(
        mh_o,
        cb_model,
        metric=metric
    )
    # Reclassify model using the optimized TH.
    y_pred, opt_precision, opt_recall, loss, conf_matrix, class_report = reclassify_model(
        mh_o,
        fitted_model,
        opt_threshold
    )
    # Print results
    print_classification_report(conf_matrix, class_report, precision,
        recall, mh_o.x_test, fitted_model, mh_o.y_test, y_pred, opt_precision,
            opt_recall, loss, opt_threshold
    )

    return fitted_model

def top_model_influencers(model, X, n_influencers=5, figsize=(10, 8)):
    importance = model.feature_importances_
    # SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    # Calculate mean SHAP values for each feature
    mean_shap_values = np.mean(shap_values.values, axis=0)
    # Create a DataFrame
    shap_df = pd.DataFrame({'Feature': X.columns, 'Score': mean_shap_values})
    # Sort by mean SHAP value
    shap_df = shap_df.sort_values(by='Score', ascending=False)
    # Create second df to store the top positive and negative influencer features
    # Exclude date related features as the dates are batch jobs entry and not actual surver date entry
    criteria = (shap_df['Feature'] != 'year') & (shap_df['Feature'] != 'month') & \
        (shap_df['Feature'] != 'day') & (shap_df['Feature'] != 'hour') & (shap_df['Feature'] != 'minute')
    tmp_df = shap_df[criteria]
    topn_pos = tmp_df.head(n_influencers)
    topn_neg = tmp_df.tail(n_influencers)
    topn_shap_df = pd.concat([topn_pos, topn_neg])
    topn_shap_df.reset_index(drop=True, inplace=True)
    # Return the top negative and positive influencers specified by n_influencers.
    return topn_shap_df

def kfold_model_cross_validation(mh_o, model, n_splits=15, random_state=random_state):
    """k-fold cross-validate trained model against test data"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Evaluate xgboost on train data
    cv_scores = cross_val_score(model, mh_o.X_train, mh_o.y_train, cv=kf, scoring='f1')

    print(f'Train data CV scores: {cv_scores}')
    print(f'Train data Mean CV score: {np.mean(cv_scores):.4f}')
    print(f'Train data CV scores std: {np.std(cv_scores):.4f}')

    # Evaluate xgboost on test data
    cv_scores = cross_val_score(model, mh_o.x_test, mh_o.y_test, cv=kf, scoring='f1')

    print(f'\nTest data CV scores: {cv_scores}')
    print(f'Test data Mean CV score: {np.mean(cv_scores):.4f}')
    print(f'Test data CV scores std: {np.std(cv_scores):.4f}')

def bootstrap_model_cross_validation(mh_o, model, n_splits, n_iterations, 
        random_state=random_state):
    """k-fold cross-validate bootstrapped samples"""
    bootstrap_cv_scores = []
    # Get the size of our training data
    n_size = len(mh_o.X)
    # Create KFold instance with input specified n_splits
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # Do bootstrap with n_iterations
    for i in range(n_iterations):
        # Pick random rows from the dataset, 
        # return random indexes from the dataset
        sample_indexes = np.random.randint(0, n_size, n_size)
        # Extract sample data from the dataset using the random indexes
        X_train = mh_o.X.iloc[sample_indexes].values
        y_train = mh_o.y.iloc[sample_indexes].values
        # Get scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='f1')
        bootstrap_cv_scores.append(cv_scores)
    
    # Convert list into pandas series
    # Calculate mean and confidence intervals
    mean_scores = pd.Series(bootstrap_cv_scores).mean()
    conf_interval = np.percentile(bootstrap_cv_scores, [2.5, 97.5])
    # Print scores
    print(f'Mean Cross-Validation Scores: {mean_scores}')
    print(f'Mean Overall Cross-Validation Score: {mean_scores.mean():.4f}')
    print(f'95% Confidence Interval: {conf_interval}')

def cramers_v(x, y):
    """Calculate Cramer's V"""
    # Create a contingency table
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(k-1, r-1))))

def ordinal_cramers_v_matrix(df):
    """Generate a matrix of CV for all categorical features"""
    #categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    categorical_cols = df.columns
    n = len(categorical_cols)
    cramers_v_matrix = pd.DataFrame(np.zeros((n, n)), index=categorical_cols, columns=categorical_cols)
    
    for i in range(n):
        for j in range(i, n):
            v = cramers_v(df[categorical_cols[i]], df[categorical_cols[j]])
            cramers_v_matrix.iloc[i, j] = v
            cramers_v_matrix.iloc[j, i] = v  # Symmetric matrix

    return cramers_v_matrix

def cramers_v_corr_to_target(df, target: str):
  """Get Cramer's V correlation to Target"""
  cramers_v_scores = {}
  features = df.drop(columns=[target])

  for feature in features:  # Skip the target column itself
      score = cramers_v(df[feature], df[target])
      cramers_v_scores[feature] = score

  # Display the results
  return pd.DataFrame(list(cramers_v_scores.items()), columns=['Feature', "CV"])

def feature_reduction(df, threshold=0.1):
  """Reduce features. Return reduced dataset"""
  selector = VarianceThreshold(threshold=threshold)
  reduced_df = selector.fit_transform(df)

  # Convert back to DataFrame (optional, for easier viewing)
  return pd.DataFrame(reduced_df, columns=df.columns[selector.get_support()])

def feature_reduction_kbest(df, target, k=10):
   # Split data into train and test sets
  X = df.drop(columns=[target], axis=0)
  y = df[target]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Initialize SelectKBest with ANOVA F-test (for classification problems)
  k_best = SelectKBest(score_func=f_classif, k=k)  # Choose top 2 features based on score

  # Fit and transform the train and test data
  X_train_selected = k_best.fit_transform(X_train, y_train)
  X_test_selected = k_best.transform(X_test)

  # Get the names of selected features
  selected_features = X.columns[k_best.get_support(indices=True)]

  # Display results
  print("Selected Features:", selected_features)
  print("Scores of selected features:", k_best.scores_[k_best.get_support(indices=True)])