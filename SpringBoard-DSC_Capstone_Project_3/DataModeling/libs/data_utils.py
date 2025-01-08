import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, RocCurveDisplay, ConfusionMatrixDisplay

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE

from sklearn.base import BaseEstimator, ClassifierMixin

# Consts

target_class_mapping = {1: '0 Days', 2: '1-13 Days', 3: '14+ Days', 9: 'Unsure'}
_MENT14D_label_mapping = {1:0, 2:1, 3:2, 9:3}
alt_target_class_mapping = {0: '0 Days', 1: '1-13 Days', 2: '14+ Days', 3: 'Unsure'}
label_mapping = _MENT14D_label_mapping

def target_label_mapping(label_map=label_mapping, y=None):
  return np.vectorize(label_map.get)(y)

# Classification Algorithms: Use in modeling. Compare performance.

class XGBCompatibleClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, **kwargs):
    self.model = XGBClassifier(**kwargs)
  
  def fit(self, X, y, **fit_params):
    self.model.fit(X, y, **fit_params)
    return self
  
  def evals_result(self):
    return self.model.evals_result()
  
  def predict(self, X):
    return self.model.predict(X)
  
  def predict_proba(self, X):
    y_proba = self.model.predict_proba(X)
    return y_proba

  def get_params(self, deep=True):
    return self.model.get_params(deep=deep)

  def set_params(self, **params):
    self.model.set_params(**params)
    return self

def create_xgb_model(**params):
  """Create instance of XGBClassifier"""
  return XGBCompatibleClassifier(
    **params,
    eval_metric='mlogloss',
    n_jobs=-1,
    random_state=42
  )

def create_lgb_model(verbose=1, **params):
  """Create instance of LightGBMClassifier"""
  return LGBMClassifier(
    **params,
    class_weight='balanced',
    n_jobs=-1, 
    random_state=42,
    verbose=verbose
  )

def create_random_forest_model():
  """Create Instance of RandomForestClassifier"""
  # Return base model with no hyperparameter tuning
  return RandomForestClassifier(
    class_weight='balanced',
    max_depth=3000,
    n_jobs=-1,
    random_state=42,
    verbose=0
  )

def create_logistic_regression_model():
  """Create Instance of LogisticRegression Model"""
  # Return base model with no hyperparameter tuning
  return LogisticRegression(
    class_weight='balanced',
    max_iter=3000,
    n_jobs=-1,
    random_state=42,
    verbose=0
  )

def evaluate_model(pipeline, X_train, x_test, y_train,  y_test, **params):
  """
  Multi-class classification Evaluation method:
  - Train model
  - Predict 
  - Plot classification charts: PR/ROC
  """
  class_names = list(target_class_mapping.values())

  model = pipeline.named_steps['classifier']
  if isinstance(model, XGBCompatibleClassifier):
    # XGB Requires target labels to be a continuous value (e.g: 0, 1, 2...).
    # target_label_mapping converts the labels defined by label_mapping
    _y_train = target_label_mapping(label_mapping, y_train)
    _y_test = target_label_mapping(label_mapping, y_test)
  else:
    _y_train = y_train
    _y_test = y_test

  ## Train model
  pipeline.fit(X_train, _y_train, **params)

  ## Model prediction
  y_pred = pipeline.predict(x_test)
  y_probs = pipeline.predict_proba(x_test)

  ## Visualize results

  # 1. Plot Confusion matrix
  cm = confusion_matrix(_y_test, y_pred)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
  _, ax = plt.subplots(figsize=(4, 3))
  disp.plot(ax=ax, cmap='viridis')
  plt.title('Confusion Matrix')
  #sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
  #  xticklabels=class_names, yticklabels=class_names)
  plt.title('Confusion Matrix')
  plt.xlabel('Predicted')
  plt.xticks(rotation=35)
  plt.ylabel('True')
  plt.show()

  # 2. Print Classification Report
  if isinstance(model, XGBCompatibleClassifier):
    # XGB Remaps the class labels to 0-indexed continues value.
    # We map the labels to the altered symbols.
    y_true_str = [alt_target_class_mapping[val] for val in _y_test]
    y_pred_str = [alt_target_class_mapping[val] for val in y_pred]
  else:
    y_true_str = [target_class_mapping[val] for val in _y_test]
    y_pred_str = [target_class_mapping[val] for val in y_pred]
  report = classification_report(y_true_str, y_pred_str, labels=class_names)
  print('Classification Report:\n')
  print(report)

  # 3. Print PR/ROC Curves
  plot_classification_stats2(_y_test, y_probs)

###
### Rev. 2
###
def evaluate_model2(pipeline, X_train, x_test, y_train,  y_test, **params):
  """
  Multi-class classification Evaluation method:
  - Train model
  - Predict 
  - Plot classification charts: PR/ROC
  """
  ## 1. Set class labels conforming to the model requirement.
  class_names = list(target_class_mapping.values())
  model = pipeline.named_steps['classifier']
  if isinstance(model, XGBCompatibleClassifier):
    # XGB Requires target labels to be a continuous value (e.g: 0, 1, 2...).
    # target_label_mapping converts the labels defined by label_mapping
    _y_train = target_label_mapping(label_mapping, y_train)
    _y_test = target_label_mapping(label_mapping, y_test)
  else:
    _y_train = y_train
    _y_test = y_test

  ## 2. Train model
  pipeline.fit(X_train, _y_train, **params)

  ## 3. Model prediction
  y_pred = pipeline.predict(x_test)
  y_probs = pipeline.predict_proba(x_test)

  ## Visualize results

  alt = False # Use alternate labeling for class names
  if isinstance(model, XGBCompatibleClassifier):
    alt = True

  # 4. Plot classification report, confusion matrix
  plot_classification_stats2(_y_test, y_pred, class_names, alt)

  # 5. Plot PR, ROC Curves
  plot_pr_roc_curve(_y_test, y_probs)
  
def plot_classification_stats(y_test, y_probs):
  """
  Plot classification stats
  """
  if isinstance(y_test, np.ndarray):
    n_classes = list(np.unique(y_test).astype(int))
  else:
    n_classes = list(y_test.unique().astype(int))

  ## Plot classification charts
  # Plot PRC and ROC

  precision = dict()
  recall = dict()
  fig, ax = plt.subplots(len(n_classes), 2, figsize=(6, 11))
  ax = ax.reshape(len(n_classes), 2)

  for i, class_n in enumerate(n_classes):
    if n_classes[0] == 0: # Starts at 0, this is XGB labeling.
      class_name = alt_target_class_mapping[class_n]
    else:
      class_name = target_class_mapping[class_n]
    # Calculate precision-recall
    precision, recall, _ = precision_recall_curve((y_test==class_n).astype(int), y_probs[:, i])
    # Calculate F1 scores to find the optimal threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)

    ax[i, 0].plot(recall, precision, marker='.')
    ax[i, 0].set_title(f'{class_name}: PR Curve')
    ax[i, 0].plot(recall[best_idx], precision[best_idx], 'ro', markersize=4, label='Threshold')
    ax[i, 0].set_xlabel('Recall')
    ax[i, 0].set_ylabel('Precision')
    ax[i, 0].grid()

    # ROC Curve
    RocCurveDisplay.from_predictions((y_test == class_n).astype(int), y_probs[:, i], ax=ax[i, 1])
    ax[i, 1].set_title(f'ROC Curve')
    ax[i, 1].grid()

  fig.tight_layout()
  _ = plt.show()

###
### Rev.
###
def plot_classification_stats2(y_test, y_pred, class_names, alt=False):

  # Create a subplot with 2 columns for confusion matrix, classification report.
  fig, ax = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={'width_ratios': [1.3, 0.7]})

  # Get confusion Matrix
  cm = confusion_matrix(y_test, y_pred)

  # Plot Confusion Matrix
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
  disp.plot(ax=ax[0], cmap='viridis', colorbar=False)
  ax[0].set_title('Confusion Matrix', fontsize=10)
  ax[0].set_xlabel('Predicted')
  ax[0].set_ylabel('True')
  ax[0].tick_params(axis='x', rotation=35)

  # Classification Report
  report = classification_report(y_test, y_pred, target_names=class_names)
  report = "Classification Report:\n\n" + report

  # Plot Classification Report
  ax[1].axis('off')  # Turn off axes for the text plot
  ax[1].text(0.1, 0.5, report, fontsize=9, fontfamily='monospace', 
    ha='left', va='center', wrap=False, transform=ax[1].transAxes)

  # Adjust layout
  plt.tight_layout()
  plt.show()

####
#### New version
####
def plot_pr_roc_curve(y_test, y_probs):
  """
  Plot classification stats
  """
  from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score

  # Binarize true labels for one-vs-rest PR and ROC analysis
  if isinstance(y_test, np.ndarray):
    classes = alt_target_class_mapping
  else:
    classes = target_class_mapping

  # Create a subplot for PR and ROC curves
  fig, ax = plt.subplots(1, 2, figsize=(8, 4))

  # Iterate through each class
  for i, (class_idx, class_label) in enumerate(classes.items()):

    # Get predictions and probabilities for this class
    y_true = (y_test==class_idx).astype(int)
    y_prob = y_probs[:, i] 

    # Compute PR Curve and Average Precision
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap_score = average_precision_score(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)

    # Optimal thresholds
    ax[0].plot(recall[best_idx], precision[best_idx], 'ko', markersize=4)
    # Plot PR Curve
    ax[0].plot(recall, precision, label=f"{class_label} (AP: {ap_score:.2f})")

    # Compute ROC Curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    ax[1].plot(fpr, tpr, label=f"{class_label} (AUC: {roc_auc:.2f})")

  # PR Curve subplot settings
  ax[0].set_title('Precision-Recall Curve', fontsize=10)
  ax[0].set_xlabel('Recall')
  ax[0].set_ylabel('Precision')
  ax[0].legend(loc="lower left")
  ax[0].grid()

  # ROC Curve subplot settings
  ax[1].set_title('ROC Curve', fontsize=10)
  ax[1].set_xlabel('False Positive Rate')
  ax[1].set_ylabel('True Positive Rate')
  ax[1].plot([0, 1], [0, 1], color='dimgray', linestyle='--')
  ax[1].legend(loc="lower right")
  ax[1].grid()

  # Layout
  plt.tight_layout()
  # Render chart
  _ = plt.show()

def train_cds_mental_health_data_model(df, target):
  """
  Perform a pipelined approach to data transformation, feature Engineering. 
  Then train, test, split the trasnddata.
  """
  ###### Group by continuous and categorical features

  # Create a disposable copy for this analysis.
  # - Drop MENTHLTH 
  df = df.drop(columns=['MENTHLTH']).copy()

  behavioral_features = ['EXEROFT1', 'CHOLCHK3']
  mental_health_features = ['EMTSUPRT', 'ADDEPEV3', 'POORHLTH']
  income_education_features = ['INCOME3', 'EDUCA']

  socioeconomic_features = ['FOODSTMP', 'SDHBILLS', 'SDHFOOD1'] # PCA candidates
  continuous_features = ['PHYSHLTH', 'POORHLTH', 'MARIJAN1'] # Numeric
  exclusive_features = socioeconomic_features + continuous_features
  categorical_features = [col for col in df.columns if col not in (exclusive_features + [target])]

  ###### Create interaction terms for:

  # 1. Socioeconomic variables using PCA for feature reduction.
  encoder = OneHotEncoder(drop='first')
  encoded = encoder.fit_transform(df[socioeconomic_features])
  scaler = StandardScaler(with_mean=False)
  scaled = scaler.fit_transform(encoded)
  socioeconomic_pca = PCA(n_components=1)
  df['Socioeconomic_Index'] = socioeconomic_pca.fit_transform(scaled)

  # 2. Behavioral and Preventive Health Variables.
  # Using Cross-product feature
  df['Behavioral_cross'] = df['EXEROFT1'].astype(int) * df['CHOLCHK3'].astype(int)

  # 3. Health Interdependencies (Polynomial Interactions)
  # Using Nonlinear interaction
  df['Physical_Mental_Interaction'] = df['GENHLTH'].astype(int) * df['PHYSHLTH'] #* df['MENTHLTH']

  # 4. Income and Education Interaction
  df['Income_Education_Interaction'] = df['INCOME3'].astype(int) * df['EDUCA'].astype(int)

  # 5. Mental Health
  df['Mental_Health_Composite'] = df[mental_health_features].mean(axis=1)

  ######  Encode categorical features and scale continuous and aggregated features

  aggregated_features = [
    'Mental_Health_Composite',
    'Income_Education_Interaction', 
    'Physical_Mental_Interaction',
    'Behavioral_cross',
    'Socioeconomic_Index'
  ]

  # Setup pre-procssing and modeling pipeline
  preprocessor = ColumnTransformer([
      ('scale', StandardScaler(), continuous_features + aggregated_features),
      ('encode', OneHotEncoder(handle_unknown='ignore'), categorical_features)
  ], remainder='passthrough', force_int_remainder_cols=False)

  model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
  ])

  ###### Prepare for model training and prediction

  # Use a small portion of the dataset for testing.
  X = df.drop(columns=socioeconomic_features + [target])
  y = df[target]

  # Split the data into train and test sets
  X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
