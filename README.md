# 💰 Investment-Worthiness-Prediction

### Machine Learning Based Investment Decision Prediction

An end-to-end **Machine Learning classification project** that predicts whether an investment opportunity is **Investment Worthy** or **Not Investment Worthy** based on financial, business, customer, risk, and market-related factors.

The project compares **Logistic Regression** and **Decision Tree** models, applies **GridSearchCV hyperparameter tuning**, evaluates model performance using multiple classification metrics, and provides an interactive pipeline for predicting new investment opportunities.

---

## 📌 Project Overview

Investors evaluate multiple factors before making an investment decision, including:

- Financial performance
- Revenue growth
- Profitability
- Debt levels
- Return on Equity
- Customer growth and retention
- Business risk
- Expected ROI
- Investment amount
- Industry growth

Manually evaluating all these factors can be time-consuming and inconsistent.

This project explores whether Machine Learning can transform these business and financial signals into a **binary investment decision**:

> **1 → Investment Worthy**  
> **0 → Not Investment Worthy**

The project follows a complete ML workflow:

**Data Loading → Data Cleaning → EDA → Feature Preparation → Train/Test Split → Model Building → Evaluation → Hyperparameter Tuning → User Input Prediction**

---

## 🎯 Problem Statement

> **Can Machine Learning predict whether an investment opportunity is worth investing in based on financial, business, and market-related factors?**

---

## 🎯 Project Objectives

The major objectives of this project are:

1. Predict whether an investment opportunity is worthy of investment.
2. Analyze the financial and business factors associated with investment decisions.
3. Build classification models using Logistic Regression and Decision Tree.
4. Evaluate models using:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - Confusion Matrix
5. Improve model performance using `GridSearchCV`.
6. Compare tuned and untuned models.
7. Accept new investment details from a user and generate a prediction.
8. Translate model results into business-oriented insights.

---

# 📊 Dataset

The project uses a synthetic investment dataset containing:

- **10,000 records**
- **19 columns**
- Target variable: `Investment_Decision`

### Target Distribution

| Investment Decision | Records |
|---|---:|
| Yes | 6,456 |
| No | 3,544 |

The target variable contains:

- `Yes` → Investment Worthy
- `No` → Not Investment Worthy

The target is encoded into numerical form for Machine Learning.

---

## 📋 Dataset Features

### Financial Features

| Feature | Description |
|---|---|
| `Annual_Revenue` | Annual company revenue |
| `Revenue_Growth_Pct` | Revenue growth percentage |
| `Profit_Margin_Pct` | Company profit margin |
| `Debt_to_Equity` | Debt-to-equity ratio |
| `Current_Ratio` | Current ratio |
| `ROE_Pct` | Return on Equity |

### Business & Customer Features

| Feature | Description |
|---|---|
| `Customer_Growth_Pct` | Customer growth percentage |
| `Customer_Retention_Pct` | Customer retention percentage |
| `Management_Experience_Years` | Management experience |
| `Business_Risk` | Business risk level |
| `Industry` | Industry category |
| `Company_Age_Years` | Company age |

### Investment & Market Features

| Feature | Description |
|---|---|
| `Expected_ROI_Pct` | Expected return on investment |
| `Investment_Amount` | Required investment |
| `Market_Share_Pct` | Company market share |
| `Market_Risk` | Market risk level |
| `Industry_Growth_Pct` | Industry growth rate |

### Target

| Feature | Description |
|---|---|
| `Investment_Decision` | Yes / No investment decision |

---

# 🔄 Machine Learning Workflow

```text
Business Problem
       ↓
Dataset Collection
       ↓
Data Understanding
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Feature Preparation
       ↓
Train-Test Split
       ↓
Model Building
       ↓
Model Evaluation
       ↓
Hyperparameter Tuning
       ↓
User Input Prediction
       ↓
Business Insights
```

---

# 🧹 Data Cleaning

The dataset was checked for:

- Missing values
- Duplicate records
- Duplicate company IDs
- Data types
- Unique values
- Descriptive statistics

Missing values were handled using **median imputation** for numerical variables.

The following columns contained missing values and were imputed:

- `Revenue_Growth_Pct`
- `Profit_Margin_Pct`
- `Current_Ratio`
- `Market_Share_Pct`
- `Customer_Retention_Pct`

After cleaning:

- Missing values → **0**
- Duplicate rows → **0**
- Company names → unique

Median imputation was selected because the median is less affected by extreme values.

---

# 📈 Exploratory Data Analysis

Several EDA visualizations were created to understand relationships between investment decisions and business factors.

### Visualizations include:

- Investment Decision Distribution
- Investment Decision by Industry
- Average Expected ROI by Investment Decision
- Average Profit Margin by Investment Decision
- Average Revenue Growth by Investment Decision
- Average Debt-to-Equity by Investment Decision
- Average Customer Retention by Investment Decision
- Investment Decision by Business Risk
- Investment Decision by Market Risk
- Correlation Heatmap

### Key EDA observations

The analysis indicates that:

- Expected ROI is an important separator between the two investment classes.
- Investment-worthy opportunities generally show higher revenue growth.
- Higher profit margins are associated with investment-worthy observations.
- Higher Debt-to-Equity is associated with less favorable investment decisions.
- Business and market risk influence the distribution of investment decisions.

---

# 🧩 Feature Selection

The model uses the following **11 features**:

```python
[
    'Industry',
    'Revenue_Growth_Pct',
    'Profit_Margin_Pct',
    'Debt_to_Equity',
    'ROE_Pct',
    'Customer_Growth_Pct',
    'Customer_Retention_Pct',
    'Business_Risk',
    'Expected_ROI_Pct',
    'Investment_Amount',
    'Industry_Growth_Pct'
]
```

The target variable is:

```python
Investment_Decision
```

`Company_Name` was excluded because it acts as a unique identifier rather than a meaningful predictive feature.

---

# 🔤 Categorical Encoding

Categorical variables were converted into numerical values using:

```python
LabelEncoder()
```

The notebook stores the encoders in a dictionary so that categorical user inputs can later be transformed before prediction.

---

# ✂️ Train-Test Split

The dataset was divided into:

- **80% Training Data**
- **20% Testing Data**

Using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This produced:

- **8,000 training records**
- **2,000 testing records**

---

# 🤖 Machine Learning Models

## 1. Logistic Regression

The first classification model was Logistic Regression.

### Base Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.80 |
| Precision | 0.82 |
| Recall | 0.89 |
| F1 Score | 0.85 |

---

## 2. Tuned Logistic Regression

Hyperparameter tuning was performed using:

```python
GridSearchCV()
```

The parameter grid included:

```python
{
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}
```

### Best Parameters

```text
C = 1
penalty = l1
solver = liblinear
```

### Tuned Performance

| Metric | Before Tuning | After Tuning |
|---|---:|---:|
| Accuracy | 0.80 | **0.85** |
| Precision | 0.82 | **0.86** |
| Recall | 0.89 | **0.90** |
| F1 Score | 0.85 | **0.88** |

The tuned Logistic Regression model showed improved performance across all four reported metrics.

---

# 🌳 Decision Tree

A Decision Tree Classifier was also developed as a second classification approach.

### Base Decision Tree Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.8130 |
| Precision | 0.8636 |
| Recall | 0.8421 |
| F1 Score | 0.8528 |

Decision Trees were selected because they can capture non-linear relationships and provide relatively interpretable decision rules.

---

# ⚙️ Tuned Decision Tree

The Decision Tree was optimized using `GridSearchCV`.

### Hyperparameters Tuned

```python
{
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

### Best Parameters

```text
criterion = gini
max_depth = 5
min_samples_split = 2
min_samples_leaf = 4
```

### Best Cross-Validation F1 Score

```text
0.8882
```

### Tuned Decision Tree Performance

| Metric | Score |
|---|---:|
| Accuracy | **0.8515** |
| Precision | **0.8535** |
| Recall | **0.9285** |
| F1 Score | **0.8894** |

---

# 🏆 Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.80 | 0.82 | 0.89 | 0.85 |
| Tuned Logistic Regression | 0.85 | 0.86 | 0.90 | 0.88 |
| Decision Tree | 0.8130 | 0.8636 | 0.8421 | 0.8528 |
| Tuned Decision Tree | **0.8515** | 0.8535 | **0.9285** | **0.8894** |

The tuned models achieved approximately **85% accuracy**, while the tuned Decision Tree produced an F1 score of approximately **0.89**.

---

# 🔮 User Input Prediction

The notebook includes an interactive prediction pipeline.

Users can enter:

```text
Industry
Revenue Growth
Profit Margin
Debt-to-Equity
ROE
Customer Growth
Customer Retention
Business Risk
Expected ROI
Investment Amount
Industry Growth
```

The input is transformed using the previously fitted categorical encoders and passed to the trained model.

The system then returns:

```text
INVESTMENT WORTHY
```

or

```text
NOT INVESTMENT WORTHY
```

along with an estimated investment-worthiness probability.

Example output format:

```text
==================================================
✅ INVESTMENT WORTHY
Investment Worthiness Probability: 73.38%
==================================================
```

---

# 💡 Business Insights

Based on the analysis performed in the project:

### 1. Expected ROI

Expected ROI is one of the strongest separators between investment decisions.

The presentation reports approximately:

```text
Investment Worthy → 17% average Expected ROI
Not Worthy        → 10.7% average Expected ROI
```

### 2. Profitability

Higher profit margins are associated with investment-worthy observations.

### 3. Revenue Growth

Investment-worthy observations show substantially higher average revenue growth than observations classified as not worthy.

### 4. Debt-to-Equity

Higher debt-to-equity is associated with less favorable investment decisions.

### 5. Business & Market Risk

Lower business and market risk are associated with a higher concentration of investment-worthy observations.

---

# 🏢 Business Recommendations

Based on the patterns identified in the project:

- Use the Decision Tree when explainability is important for non-technical stakeholders.
- Pay close attention to Expected ROI and Debt-to-Equity when screening opportunities.
- Consider business and market risk alongside financial indicators.
- Add more granular investment characteristics to improve future model performance.

---

# 🛠️ Technologies Used

### Programming Language

- Python

### Data Analysis

- Pandas

### Data Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn

### Models

- Logistic Regression
- Decision Tree Classifier

### Model Optimization

- GridSearchCV

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

# 📁 Project Structure

```text
Investment-Worthy-or-Not/
│
├── Investment_Worthy_or_not_model.ipynb
├── Investment_Worthy_Raw_Dataset_.csv
├── Investment_Model_Presentation.pptx
├── README.md
│
└── images/
    ├── investment_distribution.png
    ├── industry_analysis.png
    ├── correlation_heatmap.png
    ├── confusion_matrix.png
    └── model_comparison.png
```

---

# 🚀 How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Investment-Worthy-or-Not.git
```

## 2. Navigate to the Project Folder

```bash
cd Investment-Worthy-or-Not
```

## 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

## 5. Open

```text
Investment_Worthy_or_not_model.ipynb
```

## 6. Dataset Path

Update the dataset path in the notebook if required:

```python
df = pd.read_csv("Investment_Worthy_Raw_Dataset_.csv")
```

---

# 📌 Key Takeaways

This project demonstrates a complete Machine Learning workflow for a business decision-support problem:

```text
Raw Business Data
       ↓
Data Cleaning
       ↓
EDA
       ↓
Feature Engineering / Selection
       ↓
Classification
       ↓
Hyperparameter Optimization
       ↓
Model Evaluation
       ↓
Investment Prediction
```

The final tuned models achieved approximately:

- **Logistic Regression:** 85% Accuracy, 0.88 F1
- **Decision Tree:** 85.15% Accuracy, 0.8894 F1

The project demonstrates how financial and business indicators can be transformed into a machine-learning-based investment decision support system.

---

# 🔮 Future Improvements

Potential future improvements include:

- Add more investment-specific features.
- Test Random Forest.
- Test XGBoost.
- Compare additional classification algorithms.
- Build a Streamlit web application.
- Create an interactive investment dashboard.
- Add model explainability using SHAP.
- Monitor model performance and data drift over time.
- Improve the dataset with more granular business characteristics.

---

# ⚠️ Disclaimer

This project is an **academic/business analytics Machine Learning project** and should not be treated as financial advice.

The model provides predictions based on the patterns present in the dataset. Real investment decisions require additional due diligence, financial analysis, market research, risk assessment, and professional judgment.

---

## 👨‍💻 Author

**Mayank Vijayvargiya**

MBA – Business Analytics

### Skills Demonstrated

`Python` • `Pandas` • `Matplotlib` • `Seaborn` • `Scikit-learn` • `Machine Learning` • `EDA` • `Classification` • `Hyperparameter Tuning` • `Business Analytics`

---

## ⭐ If you found this project useful

Feel free to ⭐ **star the repository**, explore the notebook, and provide feedback.
