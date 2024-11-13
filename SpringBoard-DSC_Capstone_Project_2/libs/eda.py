import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, confusion_matrix, f1_score, classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, RocCurveDisplay
from sklearn.metrics import precision_score, recall_score, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Check and handle missing values
def handle_missing_values(df):
    if df.isnull().values.any():
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna('Unknown')
                if df[col].dtype == 'category':
                    df[col] = df[col].cat.add_categories('Unknown')
                if df[col].dtype == np.number: 
                    df[col] = df[col].fillna(df[col].mean())
    return df

# 2. Detect and plot outliers

def plot_outliers(df):
    """Plot the histogram normal and outliers of each numeric feature."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    fig, axes = plt.subplots(nrows=len(numeric_cols), figsize=(10, 5 * len(numeric_cols)))

    # Calculate Z-scores for each column
    z_scores = np.abs(stats.zscore(df))
    outliers = (z_scores >= 3).any(axis=1)  

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        # Plot normal data points
        sns.histplot(df[col], kde=True, ax=ax, label='Normal Data')
        # Overlay the outliers in red
        sns.histplot(df.loc[outliers, col], color='red', kde=True, ax=axes[i], label='Outliers', bins=15)
        ax.set_title(f'Outliers in {col}')
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# 2a. Detect and handle outliers
def handle_outliers(df):
    columns = df.select_dtypes(include=np.number).columns
    z_scores = np.abs(stats.zscore(df[columns].dropna()))
    z_scores = pd.DataFrame(z_scores, columns=columns, index=df.index)
    df = df[(z_scores < 3).all(axis=1)]
    return df

# 3. Remove duplicates
def remove_duplicates(df):
    df = df.drop_duplicates()
    return df

# 4. Data type conversion
def convert_data_types(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            if len(df[col].unique()) < df.shape[0] * 0.05:
                df[col] = df[col].astype('category')
    return df

# 5. Univariate Analysis
def univariate_analysis(df, ignore_cols=None):
    """Plot the distribution and Frequency of each feature"""
    columns = df.columns.tolist()
    # Remove columns specified by column names in ignore_cols
    if ignore_cols != None:
        if not isinstance(ignore_cols, list):
            ignore_cols = [ignore_cols]
        columns = list(set(columns) - set(ignore_cols))
    n_columns = len(columns)
    cols_per_row = 3
    rows = n_columns + 3 // cols_per_row

    fig, axes = plt.subplots(rows, ncols=cols_per_row, figsize=(20, rows * cols_per_row * 2))
    axes = axes.flatten()

    sns.set_theme(style="whitegrid") 
    sns.set_palette("colorblind")  

    for i in range(n_columns):
        ax = axes[i]
        if df[columns[i]].dtype != 'object':
            sns.histplot(df[columns[i]], ax=ax)
            ax.set_title(f'{columns[i]}')
            ax.set_ylabel('Frequency')
            ax.tick_params(axis='x', rotation=90)
        else:
            df.iloc[:, i].value_counts().plot(kind='bar')
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.subplots_adjust(wspace=0.6, hspace=0.5)
    plt.show()

# 6. Bivariate Analysis
def bivariate_analysis(df):
    """Plot the heatmap of each numeric feature"""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) > 0:
        sns.pairplot(df[numeric_cols])
        plt.show()
        sns.heatmap(df[numeric_cols].corr(), annot=True)
        plt.show()
    else:
        print('No numeric columns to analyze. Abort!')

# 7. Multivariate Analysis
def multivariate_analysis(df):
    """Create a PCA scatter plot of all numeric features"""
    num_df = df.select_dtypes(include=np.number).dropna()
    if num_df.shape[1] > 0:
        pca = PCA(n_components=2)
        pc = pca.fit_transform(num_df)
        plt.scatter(pc[:, 0], pc[:, 1])
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.title('PCA')
        plt.show()
    else:
        print('No Numeric columns to analyze. Abort!')

# 8. Feature Engineering
def feature_engineering(df, target):
    # One-hot encoding for categorical variables
    categorical_cols = df.select_dtypes(include='object').columns
    enc_df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    enc_df.columns = enc_df.columns.str.replace(' ', '_') 
    return enc_df.astype(int)

# 9. Data Transformation (Standardization & Normalization)
def data_transformation(df, ignore_cols):
    num_df = df.drop(columns=ignore_cols).select_dtypes(include=np.number)
    if num_df.shape[1] > 0:
        scaler = StandardScaler()
        df[num_df.columns] = scaler.fit_transform(num_df)
    else:
        print('No numeric columns to transform. Abort!')

    return df

# 10. Dimensionality Reduction
def dimensionality_reduction(df):
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] > 0:
        pca = PCA(n_components=5)
        pca_result = pca.fit_transform(num_df)
        return pca_result

# 11. Statistical Hypothesis Testing
def hypothesis_testing(df):
    for col in df.select_dtypes(include=np.number).columns:
        stat, p = stats.shapiro(df[col].dropna())
        print(f'{col} - Shapiro-Wilk test p-value: {p:.9f}')

# 12. Time Series Analysis (if applicable)
def time_series_analysis(df):
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.resample('M').mean().plot()
        plt.show()

# 13. Advanced Feature Selection (Lasso for feature importance)
def advanced_feature_selection(df, target):
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] > 0:
        lasso = Lasso(alpha=0.05)
        lasso.fit(num_df.drop(columns=target), num_df[target])
        print('Lasso Feature Importance:', dict(zip(df.columns, lasso.coef_)))
    else:
        print('No numeric fatures to process. Abort!')

# 14. Handling Imbalanced Data (SMOTE or other methods)
from imblearn.over_sampling import SMOTE
def handle_imbalanced_data(X, y):
    sm = SMOTE()
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

# 15. Anomaly Detection
def anomaly_detection(df):
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] > 0:
        iso = IsolationForest(contamination=0.1)
        df['anomaly'] = iso.fit_predict(num_df)
        print(df['anomaly'].value_counts())

# 16. Interaction Effects
def interaction_effects(df):
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] > 0:
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        poly_features = poly.fit_transform(num_df)
        return poly_features

# 17. Multicollinearity Check
def multicollinearity_check(df):
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] > 0:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vif_data = pd.DataFrame()
        vif_data['feature'] = df.columns
        vif_data['VIF'] = [variance_inflation_factor(num_df.values, i) for i in range(len(num_df.columns))]
        return vif_data[vif_data['VIF'] > 5]

# 18. Data Transformation (Box-Cox or Yeo-Johnson Transformations)
from sklearn.preprocessing import PowerTransformer
def box_cox_transformation(df):
    pt = PowerTransformer(method='box-cox')
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = pt.fit_transform(np.abs(df[[col]]))
    return df

# 19. AutoEDA Tools (optional)
def auto_eda_tools(df):
    import pandas_profiling
    profile = pandas_profiling.ProfileReport(df)
    profile.to_file('eda_report.html')
    print('AutoEDA report saved as eda_report.html')

# 20. Time-Series Specific Analysis (Stationarity Testing and Lag Plot)
def time_series_specific_analysis(df):
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        result = stats.adfuller(df.select_dtypes(include=np.number).dropna().iloc[:, 0])
        print('ADF Test Statistic:', result[0], 'p-value:', result[1])
        pd.plotting.lag_plot(df.select_dtypes(include=np.number).iloc[:, 0])
        plt.show()

# 21. Multicollinearity Detection (Variance Inflation Factor - VIF and Condition Index)
def multicollinearity_detection(df):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] > 0:
        vif = pd.DataFrame()
        vif['Feature'] = df.columns
        vif['VIF'] = [variance_inflation_factor(num_df.values, i) for i in range(num_df.shape[1])]
        condition_index = np.linalg.cond(num_df)
        print('Condition Index:', condition_index)
        print('VIF scores:\n', vif)
        return vif
    else:
        print('No numeric features to process. Abort!')

# 22. Causal Inference (Propensity Score Matching)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
def propensity_score_matching(df, treatment_col, outcome_col):
    X = df.drop([treatment_col, outcome_col], axis=1)
    y = df[treatment_col]
    model = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model.fit(X_train, y_train)
    propensity_scores = model.predict_proba(X_test)[:, 1]
    return propensity_scores

# 23. Feature Importance (Tree-based Feature Importance, SHAP, or LIME)
import shap
def feature_importance(df, model, target):
    X = df.drop(columns=target)
    y = df[target]
    model.fit(X, y)
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X)
    print('SHAP feature importance plot generated')

def feature_importance_check(df, target, frac=0.05):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, max_depth=2000, random_state=234)
    feature_importance(df.sample(frac=frac), model, target)

# Main pipeline function
def eda_initial_analysis(df, target=None, ignore_features=None):
    print('Starting EDA Pipeline...\n')
    
    # Step 1: Missing Values
    print('Handle Missing Values:')
    df = handle_missing_values(df)
    
    # Step 2: Outliers
    print('Handle Outliers:')
    df = handle_outliers(df)
    
    # Step 3: Duplicates
    print('Duplicates Processing:')
    df = remove_duplicates(df)
    
    # Step 4: Data Type Conversion
    print('Convert dypes')
    df = convert_data_types(df)
    
    # Step 5: Univariate Analysis
    print('Univariate Analysis:')
    univariate_analysis(df, ignore_cols=ignore_features)
    
    # Step 6: Bivariate Analysis
    print('Bivariate Analysis:')
    bivariate_analysis(df)
    
    # Step 7: Multivariate Analysis
    print('Multivariate Analysis:')
    multivariate_analysis(df)

    return df

def feature_preparation(df, target):
    """Apply feature engineering and transformation"""
    # Step 8: Data Transformation
    print('Data Transformation:')
    df = data_transformation(df, ignore_cols=target)
    # Step 9: Feature Engineering
    print('Feature Engineering:')
    return feature_engineering(df, target)

def eda_post_analysis(df, target):  
    """
    EDA Post Feat. Eng. and Transformation for Continues numeric values.
    Not suitable for categorical - nominal - variables.
    """
    # Step 10: Dimensionality Reduction
    print('Dimensionality Reduction:')
    pca_result = dimensionality_reduction(df)
    
    # Step 12: Time Series Analysis (if applicable)
    print('Time Series Analysis:')
    time_series_analysis(df)
    
    # Step 13: Advanced Feature Selection
    if target is not None:
        print('Advanced Feature Selection:')
        advanced_feature_selection(df, target)

    # Step 14: Handling Imbalanced Data (if applicable)
    vc = df[target].value_counts()
    if target is not None and vc.min() < 0.1 * len(df):
        print('Handling Imbalanced Data:')
        X_res, y_res = handle_imbalanced_data(df.drop(columns=target), df[target])
    
    # Step 15: Anomaly Detection
    print('Anomaly Detection:')
    anomaly_detection(df)
    
    # Step 16: Interaction Effects
    print('Interaction Effects:')
    poly_features = interaction_effects(df)
    
    # Step 17: Multicollinearity Check
    print('Multicollinearity Check:')
    vif = multicollinearity_check(df)
    print(vif)

    # Step 18: Box-Cox Transformation
    print('Applying Box-Cox Transformation (if needed):')
    df = box_cox_transformation(df)

    # Step 19: AutoEDA Tools (optional)
    #print('Generating AutoEDA Report (optional):')
    #auto_eda_tools(df)

    # Step 20: Time-Series Specific Analysis
    print('Time-Series Specific Analysis (if applicable):')
    time_series_specific_analysis(df)

    # Step 21: Multicollinearity Detection (VIF and Condition Index)
    print('Detecting Multicollinearity:')
    multicollinearity_detection(df)

    # Step 22: Causal Inference (if applicable)
    if target and 'treatment_col' in df.columns:
        print('Applying Propensity Score Matching:')
        propensity_scores = propensity_score_matching(df, 'treatment_col', target)

    # Step 23: Feature Importance (SHAP/LIME)
    print('Calculating Feature Importance:')
    feature_importance_check()
    
    print('EDA Pipeline Complete.')
    return df

class MentalHealthData():
    
    def __init__(self, df, target, random_state):
        import data_utils as utils

        self.df = df
        self.X = None
        self.y = None
        self.X_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.X = self.df.drop(columns=target, axis=1)
        self.y = self.df[target]

        self.X_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.3,
            random_state=random_state
        )

        #self.X_train, self.x_test, self.y_train, self.y_test, self.X, self.y = utils.train_test_split_with_duplicates(
        #    self.df,
        #    target,
        #    test_size=0.3,
        #    random_state=random_state
        #)

def model_evaluation_test(mh_o, model):
    import data_utils as utils
    
    y_test, X, y = mh_o.y_test, mh_o.X, mh_o.y

    precision, recall, opt_precision, opt_recall, y_pred = utils.get_classification_metrics(
        mh_o, 
        model
    ) 

    # Print Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print('\nConfusion Matrix (Treatment Required):')
    print(f' No: {cm[0]}')
    print(f'Yes: {cm[1]}')
    # Print Classification Report
    report = classification_report(y_test, y_pred)
    print('\nClassification Report:')
    print(report)

    # Print the cross-validation score
    custom_scorer = make_scorer(f1_score, average='weighted')
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=utils.random_state)
    cv_scores = cross_val_score(model, X, y, scoring=custom_scorer, cv=kf) 
    print(f'Cross-V Mean Score: {cv_scores.mean():.4f}')
    print(f'Cross-V Scores:\n{cv_scores}')   
    print(f'Opt. Precision: {opt_precision}')
    print(f'Opt. Recall: {opt_recall}')

    # Plot PRC and ROC
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.5))
    # Plot PR Curve
    ax[0].plot(recall, precision, marker='.')
    ax[0].set_title('Precision-Recall Curve')
    ax[0].plot(opt_recall, opt_precision, 'ro', markersize=8, label='Optimal TH')
    ax[0].legend()
    ax[0].set_xlabel('Recall')
    ax[0].set_ylabel('Precision')
    plt.grid()
    # Plot ROC Curve
    RocCurveDisplay.from_predictions(y_test, y_pred, ax=ax[1])
    ax[1].set_title('ROC Curve')
    plt.grid()
    plt.tight_layout()
    _ = plt.show()

def analyze_tradeoffs(mh_o, model):
    from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score
    from sklearn.model_selection import cross_val_predict, StratifiedKFold

    X_train, x_test, y_train, y_test = mh_o.X_train, mh_o.x_test, mh_o.y_train, mh_o.y_test

    model.fit(X_train, y_train)
    #y_pred = model.predict(x_test)
    y_probs = model.predict_proba(x_test)[:,1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    # Calculate F1 scores and find the index of the optimal threshold (maximize F1 score)
    f1_scores = []
    for p, r in zip(precision, recall):
        if (p + r) != 0:  # Avoid division by zero
            f1_scores.append(2 * (p * r) / (p + r))
        else:
            f1_scores.append(0)
    f1_scores = f1_scores[:-1]
    opt_th_idx = np.argmax(f1_scores)
    # Select thresholds around the optimal threshold, 5 thresholds on each side
    _thresholds = []
    th_range = 150 # Include 25 TH points on each side of the optimal TH.
    intervals = 30 # Skip 5 TH at a time
    # Ensure not to exceed the bounds of the threshold list
    start_idx = max(0, opt_th_idx - th_range)
    end_idx = min(len(thresholds), opt_th_idx + th_range + 1)
    # Iterate over the selected thresholds
    for i in range(start_idx, end_idx, intervals):
        threshold = thresholds[i]
        y_pred = (y_probs >= threshold).astype(int)
        # Get confusion matrix (TP, FP, FN, TN)
        _, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        # Calculate cost (example: assume cost of FP = 1 and FN = 2)
        cost = fp + 2 * fn
        # Append row to the thresholds list
        _thresholds.append([f'{threshold:.4f}', tp, fp, fn, f'{precision[i]:.4f}', f'{recall[i]:.4f}', cost])
    # Return DataFrame
    columns = ['Threshold', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'Cost']
    return pd.DataFrame(_thresholds, columns=columns)

def chi2_comparison(df1, df2):
    """
    Compare chi2 performance of 2 datasets.

    Parameters:
    ----------
    df1 - The problem dataset
    df2 - The sane dataset

    """

    from scipy.stats import chi2_contingency

    chi_square_results = {}

    for feature in df2.columns:
        # Count frequencies of each category in the duplicated and deduplicated datasets
        duplicated_counts = df1[feature].value_counts()
        deduplicated_counts = df2[feature].value_counts()

        # Create a contingency table for the Chi-Square test
        contingency_table = pd.concat([duplicated_counts, deduplicated_counts], axis=1).fillna(0)
        contingency_table.columns = ['DF1', 'DF2']

        # Perform Chi-Square test
        chi2, p, _, _ = chi2_contingency(contingency_table)

        # Store the Chi-Square statistic and p-value for each feature
        chi_square_results[feature] = {'chi2': chi2, 'p_value': p}

    # Convert results to DataFrame and sort by Chi-Square value in descending order
    return pd.DataFrame(chi_square_results).T.sort_values(by='chi2', ascending=False)