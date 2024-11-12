
**EDA Pipeline Flowchart**

1. **Start with Raw Data**  
   ↓  
 
2. **Check Data for Missing Values**
   - If missing values exist, → **Handle Missing Values**  
     - Fill with mean/median for numerical, 'Unknown' or mode for categorical  
     - Apply forward or backward fill if time-ordered  
   - If no missing values, → **Next Step**  
   ↓  

3. **Detect and Handle Outliers**  
   - Use IQR, z-score, or visualization (box plots, scatter plots) to identify outliers  
   - Remove or cap outliers based on relevance to data context  
   ↓  

4. **Check for Duplicates**
   - If duplicates exist, → **Remove Duplicates**  
   - If no duplicates, → **Next Step**  
   ↓  

5. **Data Type Conversion**  
   - Ensure data types are correct for numerical, categorical, datetime fields  
   ↓  

6. **Perform Univariate Analysis**
   - Calculate summary statistics (mean, median, range, etc.) for numerical features  
   - Plot histograms/density plots for numerical data  
   - Generate frequency distribution for categorical data  
   - Create box plots to identify spread and potential outliers  
   ↓  

7. **Perform Bivariate Analysis**
   - Generate **Correlation Matrix** for numerical variables (choose Pearson, Spearman, or Kendall as needed)  
   - Plot **Scatter Plots** to identify relationships between two numerical features  
   - Use **Heatmaps** for high-dimensional correlation visualization  
   - For categorical pairs, use **Crosstabs** or grouped **Box Plots**  
   ↓  

8. **Conduct Multivariate Analysis**
   - Apply **Pair Plots** for multiple variable relationships  
   - If high-dimensional, consider **PCA** or **Factor Analysis**  
   - Explore **Clustering** to group similar data points if relevant  
   ↓  

9. **Feature Engineering**
   - Check if new features are necessary, → **Create New Features** (e.g., feature interactions)  
   - Handle skewed data with **Log Transformation** if necessary  
   - Apply **One-Hot Encoding** for categorical variables as needed  
   - Add **Interaction Features** to capture relationships between multiple features  
   ↓  

10. **Visualize Data**
   - Use **Bar/Line Charts** for categorical trends and **Histograms** for distributions  
   - Use **Violin/KDE Plots** for richer distribution insights  
   - If dense data, use **Hexbin Plots**  
   - For text data, use **Word Clouds** to visualize frequent terms  
   ↓  

11. **Analyze Target Variable**  
   - Plot **Distribution of Target Variable** to understand classes or numerical spread  
   - **Class Imbalance**: Check if classes are balanced; if not, consider handling imbalance  
   - Examine **Target vs. Features** relationships using visualizations (e.g., bar charts, box plots)  
   ↓  

12. **Dimensionality Reduction**  
   - If dataset is high-dimensional, apply **Variance Thresholding**  
   - Use **PCA** or **t-SNE** for further reduction while preserving variance  
   ↓  

13. **Statistical Hypothesis Testing**
   - For numerical comparisons, apply **T-tests** or **ANOVA**  
   - For categorical relationships, apply **Chi-Square Test**  
   - Test distribution normality with **Shapiro-Wilk Test** or **Kolmogorov-Smirnov Test**  
   ↓  

14. **Text Data Exploration (if applicable)**  
   - Tokenize and analyze word frequency, **Sentiment Analysis** if relevant  
   - Use **N-gram Analysis** for common phrase detection  
   ↓  

15. **Time Series Analysis (if applicable)**  
   - For time series, decompose into **Trend**, **Seasonal**, **Residual** components  
   - Apply **Moving Averages** for smoothing  
   - Use **Autocorrelation** to identify seasonal patterns  
   ↓  

16. **Advanced Feature Selection**  
   - Use **RFE** or **Lasso/Ridge** for regularized feature selection  
   - Calculate **Mutual Information Scores** to assess feature relevance  
   - If working with tree-based models, consider **Boruta Algorithm** for feature selection  
   ↓  

17. **Handling Imbalanced Data (if applicable)**  
   - Apply **SMOTE** for synthetic sample generation if classes are imbalanced  
   - Consider **Oversampling/Undersampling**  
   - **Cost-sensitive Analysis** if model can handle weighted classes  
   ↓  

18. **Data Transformation**  
   - Standardize or normalize numerical features (especially for distance-based models)  
   - If data is not normally distributed, apply **Box-Cox** or **Yeo-Johnson** transformations  
   ↓  

19. **AutoEDA Tools (optional)**  
   - Use **Pandas Profiling**, **Sweetviz**, or **D-Tale** for comprehensive automated reports  
   - Consider **EDA Libraries** like `dataprep.eda`, `Lux`, or `Autoviz`  
   ↓  

20. **Check for Multicollinearity**  
   - If multicollinearity is suspected, calculate **VIF**  
   - Use **Condition Index** to identify and address multicollinearity  
   ↓  

21. **Anomaly Detection**  
   - If outliers require further scrutiny, use **Isolation Forests** or **LOF**  
   - For Gaussian-distributed data, consider **Elliptic Envelope** for anomaly detection  
   ↓  

22. **Assess Interaction Effects**  
   - Apply **Polynomial Features** to add interaction terms and polynomial terms if needed  
   - **Binning** continuous features to capture threshold effects may be helpful  
   ↓  

23. **Domain-Specific Feature Engineering**  
   - Use domain knowledge to create features or transformations specific to data context:  
     - For text, apply **TF-IDF**  
     - For images, consider resizing, normalization  
     - In healthcare, use risk scores or medical indices  
   ↓  

24. **Test for Distribution Comparisons (if applicable)**  
   - For sample comparison, apply **Kolmogorov-Smirnov** or **Anderson-Darling Tests**  
   - Use **Levene’s Test** for homogeneity of variances across groups  
   ↓  

25. **Causal Inference (if applicable)**  
   - For observational data, apply **Propensity Score Matching** to balance groups  
   - Use **Instrumental Variable Analysis** to address confounding factors  
   ↓  

26. **Evaluate Feature Importance with Dimensionality Reduction**  
   - For models, check **Tree-based Feature Importance**  
   - Use **SHAP** or **LIME** to understand feature contributions to individual predictions  
   ↓  

27. **Finalize EDA Results**  
   - Summarize insights, correlations, key features, outliers, and distributions  
   - Document EDA findings and prepare for the next stage of analysis or modeling  
