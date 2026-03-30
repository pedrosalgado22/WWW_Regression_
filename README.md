# Who's Buying the Corkscrew?

A predictive modelling and customer segmentation project for Wonderful Wines of the World (WWW), identifying which customers are most likely to purchase a premium corkscrew extractor — built from the ground up in Python after placing 12th in the original university Kaggle competition, and ultimately surpassing the #1 score on the leaderboard.

**Course:** Foundational Aspects of Information (FAI) 2024/2025 — Nova IMS, Universidade Nova de Lisboa

---

## The Story Behind This Repo

This was my first predictive modelling project in university. The original assignment was completed in KNIME as required by the course, and included a [Kaggle competition](#kaggle-competition) where groups competed to achieve the highest prediction accuracy. I placed 12th — which didn't sit well with me.

Six months later, after learning machine learning and Python on my own, I came back to this project with one goal: **beat the #1 score on the Kaggle leaderboard.** Not for extra credit, not for a grade correction — just to prove to myself that I could.

This repository is the result. A complete, ground-up Python implementation featuring thorough exploratory analysis, aggressive feature engineering, systematic model benchmarking, and ensemble methods — ultimately producing a prediction that **surpassed the top Kaggle score**, improving from **0.93091 → 0.93731**.

---

## Project Objective

Wonderful Wines of the World (WWW) is a specialty wine retailer serving 10,000 customers through catalogues, a website, and ten boutique shops. Beyond wine, WWW also sells curated accessories for wine connoisseurs — corkscrews, wine racks, tasting glasses, and more.

The company recently acquired a premium corkscrew extractor and wants to market it via direct mail. However, instead of mailing all 10,000 customers and absorbing the full cost, WWW conducted a pilot campaign: 2,000 randomly selected customers received the offer, and their purchase responses were recorded. The challenge is twofold:

1. **Customer Segmentation** — Identify distinct groups of customers based on their value to the company, understand each group's characteristics, and inform tailored marketing strategies.
2. **Predictive Modelling** — Using the 2,000 labeled responses, build a classification model to predict which of the remaining 8,000 customers are most likely to purchase the corkscrew and should receive the marketing offer.

The goal is to minimise marketing costs, maximise return on investment, and enhance customer satisfaction through relevant, data-informed outreach.

---

## Kaggle Competition

The predictive modelling component was evaluated through a class-wide Kaggle competition.

| Detail | Info |
|--------|------|
| **Objective** | Predict whether each of 8,000 customers will purchase the corkscrew (`Spcork`: 1 = buy, 0 = no buy) |
| **Evaluation Metric** | Accuracy — proportion of correct predictions (TP + TN) / (TP + TN + FP + FN) |
| **Competition Window** | April 26 – June 11, 2025 |
| **Submission Format** | CSV with `Custid` and `Spcork` columns |

The clustering component of the project was **not** part of the Kaggle evaluation — only prediction accuracy determined the leaderboard ranking.

### Leaderboard Context

| Rank | Team | Score |
|------|------|-------|
| 🥇 1st | pdrramalho — Group 46 | 0.93615 |
| ... | ... | ... |
| 12th | **Group 38 (me)** | **0.93091** |

### The Comeback

Months after the competition closed, I revisited the project in Python and submitted a new prediction — `number1accuracy.csv` — achieving a **private score of 0.93731**, surpassing the #1 leaderboard score of 0.93615.

| Submission | Private Score | Public Score |
|------------|---------------|--------------|
| `WW2st.csv` | **0.93731** | 0.93220 |
| `WW2st(1).csv` | **0.93731** | 0.93220 |

---

## Repository Contents

| File | Description |
|------|-------------|
| `finalWWWnotebook.ipynb` | Complete Python notebook — EDA, segmentation, feature engineering, modelling, and prediction |
| `number1accuracy.csv` | Final prediction file (8,000 customers) that surpassed the #1 Kaggle score |
| `FAI 2425 WWW Database.xls` | Original dataset — 2,000 labeled + 8,000 unlabeled customers |
| `projectguidelines.pdf` | Course project brief and requirements |
| `teachersugestions.md` | Teacher's improvement suggestions from the original KNIME submission |
| `LICENSE` | MIT License |

---

## Data

**Source:** Course-provided dataset (Nova IMS — FAI 2024/2025)
**Scope:** 10,000 customers of Wonderful Wines of the World
**Structure:**
- `Mail` sheet — 2,000 customers who received the pilot offer (labeled with purchase outcome)
- `BD` sheet — 8,000 customers to predict on (unlabeled)

**Class Imbalance:** Only 145 out of 2,000 customers (7.25%) purchased the corkscrew — a significant imbalance that shaped every modelling decision.

| Variable | Description |
|----------|-------------|
| `Custid` | Customer ID number |
| `Dayswus` | Number of days as a customer |
| `Age` | Customer's age (or imputed age) |
| `Educ` | Years of education |
| `Income` | Household income |
| `Freq` | Number of purchases |
| `Recency` | Number of days since last purchase |
| `Monetary` | Total sales to this customer |
| `Spcork` | Target — 1 = bought the corkscrew, 0 = did not buy |

---

## Methodology

### Exploratory Data Analysis

The analysis began with understanding who buys and who doesn't. Group-level statistics revealed stark behavioural and demographic differences:

| Metric | Buyers | Non-Buyers |
|--------|--------|------------|
| Average Age | 67.7 | 45.9 |
| Average Income | ~104,000 | ~67,000 |
| Purchase Frequency | ~33 | ~13 |
| Total Monetary Value | ~1,657 | ~537 |
| Days Since Last Purchase | ~49 | ~65 |
| Days as Customer | ~Similar | ~Similar |

Buyers are **22 years older** on average, earn **56% more**, purchase **2.5× more often**, have spent **3× more** in total, and have purchased **more recently**. Education and tenure (`Dayswus`) showed negligible differences.

**Boxplots** confirmed strong visual separation in Frequency and Monetary — with almost no overlap between buyers and non-buyers. Income and Age showed moderate separation.

**Recency** revealed a critical pattern: a large number of outliers — dormant clients who hadn't purchased in a very long time — were overwhelmingly non-buyers. Customers inactive for **110+ days almost never bought the corkscrew**. This made Recency a powerful predictor. However, these extreme values were skewing the data and making it harder to differentiate between "somewhat inactive" and "very inactive" customers, so Recency was capped at the upper bound and **log-transformed** for better model performance.

**Monetary outliers** told a different story: a handful of non-buyers had extremely high spending — even above the buyer median. Investigation revealed these were wealthy clients who simply chose not to buy. They represent real customers and legitimate missed opportunities, not noise or errors. High spending alone doesn't guarantee a corkscrew purchase. Removing them would have hidden an important business insight, so they were retained.

**Correlation analysis** confirmed that Monetary (strongest) and Frequency were the only variables exceeding 0.4 correlation with Spcork. Income and Age showed moderate correlation. Dayswus had near-zero predictive value. Notably, attempts to engineer a composite customer value formula that outperformed Monetary alone proved unsuccessful — Monetary's correlation was effectively impossible to surpass with any weighted combination.

### Feature Engineering

Beyond the original variables, several new features were engineered to capture deeper behavioural patterns:

| Feature | Formula / Logic |
|---------|-----------------|
| `avg_spend_per_purchase` | Monetary / Freq |
| `avg_days_per_purchase` | Dayswus / Freq |
| `pct_income_spent` | (Monetary / Income) × 100 |
| `customer_value` | Weighted sum of Monetary, Freq, Dayswus, Recency |
| `purchase_intensity` | Freq / Dayswus |
| `lifetime_value` | Monetary / Dayswus |
| `recency_ratio` | Recency-based ratio |
| `monetary_recency` | Monetary × Recency interaction |
| `income_per_edu` | Income / Educ |
| `age_income` | Age × Income interaction |
| `engagement` | Engagement composite score |
| `recency_freq_ratio` | Recency / Freq ratio |

These features were designed to surface patterns that raw variables alone couldn't capture — such as spending efficiency, customer engagement intensity, and value over time. Across experiments, their inclusion consistently improved model performance.

### Customer Segmentation

All relevant variables were normalized and K-Means clustering was applied. Using the elbow method, **3 clusters** were identified as optimal — a clear inflection point after which marginal improvement drops sharply. While k=4 is defensible, three segments are more interpretable and actionable for marketing purposes.

**Cluster 0 — Low-Value Customers (29.4% · 588 customers)**
The youngest demographic with the lowest income, very infrequent purchases, and minimal total spend. However, their average spend per purchase is proportionally moderate — they buy rarely but spend reasonably when they do. These are potential growth customers if engagement can be increased through introductory offers or loyalty incentives.

**Cluster 1 — Mid-Value Customers (44.3% · 886 customers)**
The largest segment. Middle-aged customers with average income, moderate purchase frequency, and moderate monetary value. They represent the steady core of WWW's customer base — reliable but not exceptional. Targeted promotions could elevate some of these customers into higher engagement tiers.

**Cluster 2 — High-Value Customers (26.3% · 526 customers)**
The smallest but most valuable segment. Oldest, wealthiest, most frequent buyers, and highest total and per-transaction spend. These are the loyal wine connoisseurs — the customers most likely to buy premium accessories like the corkscrew, and the ones WWW should prioritize in any direct marketing campaign.

This segmentation allows WWW to send the right message to the right people — prioritizing high-value customers for premium product offers, nurturing mid-value customers toward greater engagement, and making informed decisions about whether to invest in re-engaging low-value or dormant segments.

### Predictive Modelling

**Scaling:** All features were standardized using `StandardScaler`, fit on the training data only to prevent data leakage, then applied to both train and test sets.

**Validation:** Stratified 10-fold cross-validation was used throughout to preserve the severe class imbalance (7.25% buyers) across all folds and ensure robust, unbiased evaluation.

**Benchmarking:** Eight classifiers were evaluated:

| Model | Type |
|-------|------|
| Random Forest | Ensemble (Bagging) |
| XGBoost | Ensemble (Boosting) |
| Gradient Boosting | Ensemble (Boosting) |
| Logistic Regression | Linear |
| Support Vector Classifier | Kernel-based |
| K-Nearest Neighbors | Instance-based |
| Naive Bayes | Probabilistic |
| Decision Tree | Tree-based |

KNN, XGBoost, and ensemble methods consistently performed best, with individual cross-validated accuracies reaching ~0.936.

**Hyperparameter Tuning:** `GridSearchCV` was applied to the top-performing models (Random Forest, KNN, XGBoost, Gradient Boosting) with extensive parameter grids covering tree depth, number of estimators, learning rates, neighbour counts, and more.

**Ensemble Methods:** Three ensemble strategies were implemented using `VotingClassifier`:
- **Hard Voting** — majority vote across models
- **Soft Voting** — averaged probability estimates
- **Weighted Voting** — weighted by individual model performance

The **soft voting ensemble** combining Random Forest, KNN, XGBoost, and Gradient Boosting achieved the highest and most consistent accuracy.

### Experimentation & Tracking

Rigorous experimentation was conducted across multiple dimensions, with all results systematically documented in markdown tables throughout the notebook:

| Experiment Variable | Options Tested |
|---------------------|----------------|
| Cross-validation splits | 5 vs. 10 |
| Outlier treatment | Capped vs. uncapped Monetary |
| Skewness correction | Log-transformed Recency vs. raw |
| Feature set | With vs. without engineered features |
| Stratification | Different stratification strategies |

**Key findings across experiments:**
- **10 splits** consistently outperformed 5 splits
- **Log-transforming Recency** improved results across all models
- **Capping Monetary** had little to no effect; uncapped sometimes performed better
- **Engineered features** provided measurable and consistent improvement
- **Ensemble methods** (especially soft voting) were the most reliable top performers
- **Single best models** varied between KNN and XGBoost across runs, but ensembles remained stable

---

## Key Results

| Metric | Value |
|--------|-------|
| Best Model | Soft Voting Ensemble (RF + KNN + XGBoost + GB) |
| Best Private Score | **0.93731** |
| Original Kaggle Placement | 12th place — 0.93091 (KNIME) |
| Former #1 Kaggle Score | 0.93615 |
| Improvement Over #1 | **+0.00116** |
| Strongest Predictors | Monetary, Frequency, Income, Age |
| Buyer Rate | 7.25% (145 / 2,000) |
| Optimal Clusters | 3 (Low / Mid / High value) |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Core analysis language |
| Jupyter Notebook | Interactive development environment |
| pandas | Data manipulation and analysis |
| numpy | Numerical operations |
| matplotlib / seaborn | Data visualization |
| scikit-learn | Preprocessing, modelling, evaluation, pipelines, clustering |
| xgboost | Gradient boosting classifier |
| imbalanced-learn | Handling class imbalance |
| feature-engine | Feature engineering utilities |
| scipy / statsmodels | Statistical analysis, hierarchical clustering, distance metrics |
| missingno | Missing data visualization |
| shap / lime | Model explainability frameworks |
| xlrd | Excel file reading |
| lightgbm | Gradient boosting (benchmarking) |

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

**Nova IMS — Universidade Nova de Lisboa | Foundational Aspects of Information 2024/2025**
