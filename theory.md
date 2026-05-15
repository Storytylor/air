# Artificial Intelligence in Robotics — Theory Answer Bank
**T.E (2019) | Wadia College of Engineering, Pune**
**Subject: Artificial Intelligence in Robot**

---

## Q.3 — Data Acquisition Process for a Machine Learning Pipeline

Data acquisition is the **first and most critical step** in any machine learning pipeline. It refers to the process of collecting raw data from various sources so that it can be used to train a model.

### Steps in Data Acquisition:

1. **Problem Definition:** Before collecting data, the problem must be clearly defined — what are we predicting? What kind of output is expected (classification, regression)?

2. **Identifying Data Sources:** Data can come from:
   - Public datasets (Kaggle, UCI ML Repository, government portals)
   - Web scraping (extracting data from websites)
   - Sensors and IoT devices (especially in robotics)
   - Databases and APIs
   - Manual data collection (surveys, experiments)

3. **Data Collection:** Gathering raw data from the identified sources. In robotics, this may involve reading from cameras, LIDAR, encoders, etc.

4. **Data Storage:** Storing the collected data in a structured format — CSV files, SQL databases, JSON files, or cloud storage (AWS S3, Google Cloud).

5. **Data Quality Check:** Verifying that the collected data is:
   - Sufficient in quantity
   - Representative of all real-world scenarios
   - Free from obvious errors or corruption

6. **Data Labelling (for Supervised Learning):** If the task requires labelled data, this step involves annotating the data — for example, labelling images as "cat" or "dog."

### Importance:
The quality and quantity of data directly impact the performance of a machine learning model. Garbage in = Garbage out. Even the best algorithms fail if the data acquisition step is poorly done.

---

## Q.4 — Handling Missing Values

### What are Missing Values?
Missing values occur when no data is stored for a particular variable in an observation. They appear as `NaN` (Not a Number) or `None` in a Pandas DataFrame. Missing data can arise due to data entry errors, equipment failures, or incomplete surveys.

### Why Handle Them?
Most ML algorithms cannot process `NaN` values. If not handled, they cause errors during model training and reduce accuracy.

### Two Main Strategies:

#### Strategy 1: Dropping Rows/Columns
- **Drop rows** with missing values using `dropna()` — suitable when the proportion of missing data is very small (e.g., < 5%) and losing those rows won't affect the dataset significantly.
- **Drop columns** when a column has too many missing values (e.g., > 50%) and cannot be reliably imputed.
- **Disadvantage:** Loss of data, especially problematic for small datasets.

#### Strategy 2: Imputation (Mean / Median)
- **Mean Imputation:** Replace missing values with the arithmetic mean of that column. Best used for normally distributed (symmetric) data without many outliers.
- **Median Imputation:** Replace missing values with the median. Preferred when the data is skewed or contains outliers, since the median is more robust.
- **Mode Imputation:** For categorical columns, missing values are replaced with the most frequent value (mode).

### When to Use Which?
| Strategy | Use When |
|---|---|
| Drop rows | Missing data is random and very small in amount |
| Drop columns | Column is mostly empty or irrelevant |
| Mean imputation | Data is normally distributed, no outliers |
| Median imputation | Data is skewed or has outliers |

---

## Q.5 — Outlier Detection and Removal using IQR

### What is an Outlier?
An outlier is a data point that is **significantly different** from the rest of the dataset. Outliers can be caused by measurement errors, data entry mistakes, or genuine rare events. They distort statistical measures like the mean and negatively affect ML model performance.

### IQR (Interquartile Range) Method:
The IQR is the range between the **1st quartile (Q1)** and the **3rd quartile (Q3)** of the data.

```
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Any data point below the **Lower Bound** or above the **Upper Bound** is considered an outlier and is removed.

### Why 1.5 × IQR?
This multiplier (1.5) was proposed by statistician John Tukey. It provides a balance — strict enough to catch real outliers, but lenient enough not to remove valid extreme values.

### Effect on Data Distribution:
- **Before removal:** The boxplot shows whiskers extending far out, with dots representing outlier points.
- **After removal:** The boxplot is more compact, the spread is tighter, and statistical measures like mean and standard deviation become more representative.

### Advantages of IQR:
- Robust to extreme values (unlike Z-score which uses mean)
- Works well for skewed distributions
- Simple and interpretable

---

## Q.6 — Skewness and Transformations

### Definition of Skewness:
Skewness is a measure of the **asymmetry** of a probability distribution around its mean.

- **Positive Skew (Right Skew):** The tail is on the right side. Mean > Median > Mode. Most data is clustered on the left.
- **Negative Skew (Left Skew):** The tail is on the left side. Mean < Median < Mode. Most data is clustered on the right.
- **Zero Skew:** Perfectly symmetric distribution (e.g., normal distribution).

A skewness value between **-0.5 and 0.5** is considered approximately symmetric. Values beyond **±1** indicate high skewness.

### Why is Skewness a Problem?
Many ML algorithms (especially Linear Regression, Logistic Regression) assume that input features follow a **normal distribution**. Highly skewed features violate this assumption and degrade model performance.

### Transformations to Fix Skewness:

| Transformation | Formula | Best For |
|---|---|---|
| **Log Transformation** | `log(x)` or `log(x+1)` | Positively skewed data with no zeros |
| **Square Root Transformation** | `√x` | Moderate positive skew, non-negative data |
| **Box-Cox Transformation** | Parametric, finds best λ | Positive data; automatically finds optimal transform |

### Key Point:
- Log and square root transformations **compress large values**, reducing the effect of extreme high values in positively skewed data.
- Box-Cox is the most flexible — it finds the optimal power transformation automatically using maximum likelihood estimation.
- After transformation, the distribution should resemble a **bell curve (normal distribution)**.

---

## Q.7 — Handling Categorical Data

### What is Categorical Data?
Categorical data represents **discrete groups or labels** rather than numeric quantities. Examples: Gender (Male/Female), City (Mumbai/Pune/Delhi), Grade (A/B/C).

ML algorithms work with **numbers**, so categorical data must be converted to numerical form.

### Two Main Types of Categorical Variables:
1. **Nominal:** No natural order (e.g., Color — Red, Blue, Green)
2. **Ordinal:** Has a natural order (e.g., Education — High School < Bachelor's < Master's)

### Encoding Techniques:

#### 1. Label Encoding
- Assigns an integer to each category: Red=0, Blue=1, Green=2
- **Use When:** The variable is **ordinal** (has a natural order)
- **Problem:** For nominal data, introduces unintended ordinal relationships (e.g., the model may assume Blue > Red, which is meaningless)

#### 2. One-Hot Encoding (OHE)
- Creates a new **binary column** for each category (0 or 1)
- **Use When:** The variable is **nominal** (no natural order)
- **Problem:** Can cause the "dummy variable trap" — high dimensionality when there are many unique categories. Solved by dropping one column (`drop_first=True`)

#### 3. Ordinal Encoding
- Manually maps categories to ordered integers based on domain knowledge
- **Use When:** The variable is ordinal and you know the correct order

#### When to Use Which?
| Technique | Use Case |
|---|---|
| Label Encoding | Ordinal data, tree-based models |
| One-Hot Encoding | Nominal data, linear/logistic models |
| Ordinal Encoding | When you explicitly know the order |

---

## Q.10 — Feature Engineering and StandardScaler

### Feature Engineering:
Feature engineering is the process of **using domain knowledge to create, transform, or select input features** that make ML models more accurate and efficient. It is one of the most impactful steps in the ML pipeline.

Examples of feature engineering:
- Creating a "Age Group" column from a raw "Age" column
- Extracting "Day of Week" from a "Date" column
- Computing "BMI" from "Height" and "Weight"
- Creating interaction features (e.g., `Price × Quantity = Revenue`)

Good feature engineering can dramatically improve model accuracy even without changing the algorithm.

### Role of StandardScaler (Feature Scaling):
StandardScaler transforms features so that they have a **mean of 0** and a **standard deviation of 1**. This is called **Z-score normalization**.

```
z = (x - mean) / standard_deviation
```

### Why is Scaling Needed?
- Features can have **vastly different scales** (e.g., Age: 18–80 vs. Income: 20,000–500,000)
- Distance-based algorithms (KNN, SVM) and gradient descent-based algorithms (Linear Regression, Neural Networks) are **sensitive to feature scale**
- Without scaling, features with larger magnitudes **dominate** the model

### Why Apply Scaling AFTER Splitting the Data?
This is a critical concept to avoid **data leakage**:
- If you scale before splitting, the scaler learns the mean and standard deviation of the **entire dataset** (including the test set)
- This causes the test set statistics to "leak" into the training process, giving overly optimistic evaluation results
- **Correct approach:** Fit the scaler **only on training data**, then use that fitted scaler to transform both training and test data separately

---

## Q.11 — Regression Models: Theory

### What is Regression?
Regression is a type of **supervised learning** where the goal is to predict a **continuous numerical output** (e.g., house price, temperature, salary).

### Linear Regression:
Models the relationship between the independent variable(s) `X` and the dependent variable `y` as a **straight line**:

```
y = β₀ + β₁X + ε
```
Where:
- `β₀` = intercept (value of y when X=0)
- `β₁` = slope (change in y per unit change in X)
- `ε` = error term

The model minimizes the **Sum of Squared Errors (SSE)** using the **Ordinary Least Squares (OLS)** method.

### Polynomial Regression:
Used when the relationship is **non-linear**. It adds polynomial terms (X², X³, etc.):
```
y = β₀ + β₁X + β₂X² + ... + βnXⁿ
```

### Evaluation Metrics:

| Metric | Formula | What It Means |
|---|---|---|
| **MAE** | Mean of \|actual - predicted\| | Average absolute error. Easy to interpret. Robust to outliers. |
| **MSE** | Mean of (actual - predicted)² | Penalizes large errors more heavily due to squaring. |
| **R² Score** | 1 - (SS_res / SS_tot) | Proportion of variance explained by the model. Ranges 0 to 1; closer to 1 is better. |

---

## Q.12 — Regression vs. Classification Models

### Comparison:

| Aspect | Regression | Classification |
|---|---|---|
| **Output** | Continuous value (e.g., 25.3, 100.7) | Discrete class/label (e.g., Yes/No, Cat/Dog) |
| **Example** | Predicting house price | Detecting spam email |
| **Algorithms** | Linear Regression, Polynomial Regression | Logistic Regression, KNN, SVM, Decision Tree |
| **Loss Function** | MSE, MAE | Cross-Entropy, Hinge Loss |

### Performance Evaluation — Regression:
1. **MAE (Mean Absolute Error):** Average of absolute differences between predicted and actual values. Easy to interpret in the same units as the target variable.
2. **R² (R-Squared):** Measures how well the model explains the variance in the data. A value of 1 = perfect fit; 0 = model is no better than the mean.

### Performance Evaluation — Classification:
1. **Accuracy:** Percentage of correct predictions out of total predictions. Good for balanced datasets.
2. **F1-Score:** Harmonic mean of Precision and Recall. Useful when there is class imbalance.

### When to Choose Regression Over Classification?
Choose **Regression** when:
- The target variable is **continuous** (temperature, price, age)
- You need to **quantify** the output, not just categorize it
- The problem involves **forecasting or estimation** (e.g., stock price prediction)

Choose **Classification** when the output is a **category or class** (yes/no, type A/B/C).

---

## Q.13 — Markov Process and Markov Chain

### Markov Process:
A **Markov Process** (also called a Markov Chain in discrete settings) is a **stochastic (random) process** where the future state of the system depends **only on the current state**, and NOT on the history of past states.

### The Markov Property:
> *"The future is independent of the past, given the present."*

Formally:
```
P(Xₙ₊₁ = s | X₀, X₁, ..., Xₙ) = P(Xₙ₊₁ = s | Xₙ)
```

This "memoryless" property greatly simplifies computation in probabilistic models.

### Key Concepts:

- **State:** A particular situation the system can be in (e.g., Sunny, Rainy, Cloudy)
- **State Space:** The set of all possible states
- **Transition Probability:** The probability of moving from one state to another
- **Transition Matrix:** A matrix where entry `[i][j]` represents the probability of moving from state `i` to state `j`. Each row must sum to 1.

### Example (Weather):
If today is Sunny, there's a 70% chance of Sunny tomorrow and a 30% chance of Rainy.
This is captured in the **transition matrix**.

### Applications in AI/Robotics:
- **Reinforcement Learning** (Markov Decision Processes, MDPs)
- **Speech Recognition** (Hidden Markov Models)
- **Robot Navigation** (planning paths under uncertainty)
- **Natural Language Processing** (text generation)

---

## Q.14 — End-to-End ML Pipeline (Theory)

### Overview:
A Machine Learning pipeline is a **systematic sequence of steps** to go from raw data to a deployed model. Each step builds on the previous.

### Step-by-Step Explanation:

1. **Reading of Data:** Load the dataset into a Pandas DataFrame from a CSV/Excel/database. Inspect shape, columns, and sample rows.

2. **Handling Missing Values:** Identify `NaN` values. Apply dropping or imputation strategies (mean, median, mode) as appropriate for each column.

3. **Handling Outliers:** Use the IQR method or Z-score to detect and remove/cap extreme values. Visualize using boxplots before and after.

4. **Skewness Removal:** Check skewness of numerical features. Apply log, square root, or Box-Cox transformations to bring distributions closer to normal.

5. **Handling Categorical Data:** Apply Label Encoding (ordinal) or One-Hot Encoding (nominal) to convert text categories into numeric form.

6. **Feature Engineering:** Create new meaningful features, drop irrelevant ones, and extract useful patterns (e.g., date decomposition, interaction terms).

7. **StandardScaler:** Normalize all numerical features to zero mean and unit variance. Fit on training data only.

8. **Splitting of Data:** Divide the dataset into training set (typically 80%) and test set (20%) using `train_test_split`. This ensures unbiased model evaluation.

9. **Importing and Training a Model:** Select a suitable ML algorithm (Linear Regression, KNN, Decision Tree, etc.) and fit it on the training data.

10. **Model Evaluation:** Use appropriate metrics (MAE, MSE, R² for regression; Accuracy, F1, Confusion Matrix for classification) to evaluate on the test set.

### Why the Order Matters:
- Scaling must come **after splitting** to prevent data leakage
- Encoding must come **before training** since models need numbers
- Outlier removal must come **before imputation** to avoid imputing from corrupt data

---

## Q.15 — KNN Classification (Theory)

### What is KNN?
K-Nearest Neighbors (KNN) is a **simple, non-parametric, lazy learning algorithm** used for both classification and regression. It makes predictions based on the **K closest data points** in the training set.

### How it Works (Classification):
1. Choose the value of K (number of neighbors)
2. For a new data point, calculate its **distance** (usually Euclidean) to all training points
3. Select the K nearest neighbors
4. Assign the **majority class** among those K neighbors to the new point

### Key Concepts:
- **Lazy Learner:** KNN does not build an explicit model during training — it memorizes the training data and computes at prediction time
- **Distance Metric:** Euclidean distance is most common: `d = √(Σ(xᵢ - yᵢ)²)`
- **Choosing K:** Small K = high variance (overfitting); Large K = high bias (underfitting). Optimal K is found using cross-validation.

### Effect of Scaling on KNN:
KNN is **highly sensitive to feature scale**. Features with larger ranges dominate distance calculations. **StandardScaler must be applied before KNN.**

---

## Q.15 (Second) — Supervised vs. Unsupervised ML

### Supervised Machine Learning:
In supervised learning, the model is trained on **labelled data** — meaning each training example has an input `X` and a known output `y`.

- The model learns to **map inputs to outputs**
- During training, it minimizes the error between predicted and actual labels
- **Examples:** Linear Regression, Logistic Regression, KNN, SVM, Decision Trees
- **Use Cases:** Spam detection, house price prediction, disease diagnosis

### Unsupervised Machine Learning:
In unsupervised learning, the model is trained on **unlabelled data** — there is no output variable `y`. The model must find **hidden patterns or structures** on its own.

- The model groups or compresses data without predefined categories
- **Examples:** K-Means Clustering, DBSCAN, PCA, Autoencoders
- **Use Cases:** Customer segmentation, anomaly detection, dimensionality reduction

### Comparison:

| Aspect | Supervised | Unsupervised |
|---|---|---|
| **Data** | Labelled (X, y) | Unlabelled (X only) |
| **Goal** | Predict output | Find hidden structure |
| **Output** | Prediction / Classification | Clusters / Compressed features |
| **Evaluation** | Accuracy, MSE, F1 | Silhouette Score, Inertia |
| **Example Algorithm** | Linear Regression, KNN | K-Means, PCA |

---

## Q.16 — Data Types in Python

Python has the following built-in data types:

### 1. Numeric Types:
- **int:** Integer values — whole numbers, positive or negative. E.g., `5`, `-100`
- **float:** Decimal/floating-point numbers. E.g., `3.14`, `-0.001`
- **complex:** Numbers with real and imaginary parts. E.g., `3+4j`

### 2. Sequence Types:
- **str (String):** Ordered sequence of characters, immutable. E.g., `"Hello"`
- **list:** Ordered, mutable collection of items. E.g., `[1, 2, 3]`
- **tuple:** Ordered, immutable collection. E.g., `(1, 2, 3)`
- **range:** Immutable sequence of numbers. E.g., `range(0, 10)`

### 3. Mapping Type:
- **dict (Dictionary):** Collection of key-value pairs, unordered, mutable. E.g., `{"name": "Alice", "age": 25}`

### 4. Set Types:
- **set:** Unordered collection of **unique** elements, mutable. E.g., `{1, 2, 3}`
- **frozenset:** Immutable version of set

### 5. Boolean Type:
- **bool:** Represents `True` or `False`; subclass of int (`True = 1`, `False = 0`)

### 6. None Type:
- **NoneType:** Represents the absence of a value (`None`)

### In the Context of ML / Data Science:
Pandas extends these types to include:
- `int64`, `float64` for numerical data
- `object` for strings/mixed
- `category` for categorical variables
- `datetime64` for timestamps
- `bool` for binary features

---

## Q.17 — Assumptions of Linear Regression

Linear Regression is a powerful model, but it makes several key **statistical assumptions**. Violating these assumptions leads to unreliable results.

### The 5 Key Assumptions:

#### 1. Linearity
- The relationship between the independent variable(s) `X` and the dependent variable `y` must be **linear**
- Verified using: scatter plot of X vs. y, or residual plot
- Violation fix: Apply polynomial regression or transformations

#### 2. Independence of Errors (No Autocorrelation)
- The **residuals (errors)** should be independent of each other — no pattern across observations
- Important in time-series data (yesterday's error should not predict today's error)
- Verified using: **Durbin-Watson test** (value close to 2 = no autocorrelation)
- Violation fix: Use time-series models (ARIMA)

#### 3. Homoscedasticity (Constant Variance of Errors)
- The variance of residuals should be **constant** across all levels of X
- If variance increases/decreases with X, this is called **heteroscedasticity**
- Verified using: Residual plot (should show random scatter, not a funnel shape)
- Violation fix: Log transformation of `y`, or use Weighted Least Squares

#### 4. Normality of Errors
- The residuals should follow a **normal distribution** (bell curve)
- This is needed for valid hypothesis tests and confidence intervals
- Verified using: **Q-Q plot** (residuals should lie along the diagonal), **Shapiro-Wilk test**
- Violation fix: Transform target variable or use robust regression

#### 5. No Multicollinearity
- The independent variables should not be **highly correlated with each other**
- If two features are highly correlated (e.g., `Height_cm` and `Height_inch`), the model cannot reliably estimate individual coefficients
- Verified using: **Variance Inflation Factor (VIF)** — VIF > 10 indicates high multicollinearity
- Violation fix: Remove or combine correlated features, use Ridge Regression

### Summary Table:
| Assumption | Test / Visual | Fix |
|---|---|---|
| Linearity | Scatter plot | Polynomial features |
| Independence | Durbin-Watson test | Time series models |
| Homoscedasticity | Residual plot | Log transform target |
| Normality of errors | Q-Q plot | Transform y |
| No Multicollinearity | VIF scores | Drop features, Ridge |

---

## Q.18 — Evaluation Parameters of Classification Models

### 1. Confusion Matrix
A confusion matrix is a **table that summarizes the performance** of a classification model by comparing predicted labels against actual labels.

```
                  Predicted: Yes    Predicted: No
Actual: Yes   |  True Positive (TP)  | False Negative (FN) |
Actual: No    |  False Positive (FP) | True Negative (TN)  |
```

- **TP (True Positive):** Correctly predicted as positive
- **TN (True Negative):** Correctly predicted as negative
- **FP (False Positive):** Incorrectly predicted as positive (Type I Error)
- **FN (False Negative):** Incorrectly predicted as negative (Type II Error)

---

### 2. Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- Percentage of **correct predictions** out of all predictions
- **Good for balanced datasets**; misleading for imbalanced ones (e.g., 99% accuracy if model always predicts majority class)

---

### 3. Precision
```
Precision = TP / (TP + FP)
```
- Out of all **predicted positives**, how many were actually positive?
- Important when the **cost of False Positives is high** (e.g., spam filter — you don't want to mark legitimate emails as spam)

---

### 4. Recall (Sensitivity)
```
Recall = TP / (TP + FN)
```
- Out of all **actual positives**, how many did the model correctly identify?
- Important when the **cost of False Negatives is high** (e.g., cancer detection — missing a positive case is dangerous)

---

### 5. F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- **Harmonic mean** of Precision and Recall
- Best metric when there is a **class imbalance** or when both FP and FN matter
- Ranges from 0 (worst) to 1 (best)

---

### Trade-off Between Precision and Recall:
- Increasing the classification threshold → Higher Precision, Lower Recall
- Decreasing the threshold → Lower Precision, Higher Recall
- **F1-Score** balances the two

### Summary:

| Metric | Focus | Best Used When |
|---|---|---|
| Accuracy | Overall correctness | Balanced classes |
| Precision | Minimizing false positives | Spam detection, fraud |
| Recall | Minimizing false negatives | Disease diagnosis |
| F1-Score | Balance of precision & recall | Imbalanced datasets |
| Confusion Matrix | Full breakdown | Always — for detailed insight |

---

*End of Theory Answer Bank*
*Questions 1, 2, 8, 9 excluded as per instruction (code-only/PCA questions)*
