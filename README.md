# Module 17: Practical Application 3: Comparing-Classifiers

## Overview

This project applies multiple machine learning classifiers to predict whether a client of a Portuguese bank will subscribe to a term deposit. The primary objective is to compare different classification algorithms—K-Nearest Neighbors, Logistic Regression, Decision Trees, and Support Vector Machines—and determine which performs best on this imbalanced dataset.

Dataset
* Source:  UC Irvine Machine Learning Repository
* Size: 41,188 records × 21 features  
* Target Variable: (`y`): whether the client subscribed to a term deposit (`yes`/`no`)
* Class imbalance: ~88% “no” vs. 12% “yes” 

## Notebooks
- promptt.ipynb: [link to colab notebook code](https://drive.google.com/file/d/1fuCoWHbFDWMGShT5j53yzm_FiJa7bJH0/view?usp=sharing)

## Data Description 
Feature Groups

Client Information (Demographics)
- age: numeric
- job: type of job (e.g., admin., management, student, retired)
- marital: marital status (single, married, divorced/unknown)
- education: education level (basic, high school, university, professional course, etc.)
- default: has credit in default (yes, no, unknown)
- housing: has housing loan (yes, no, unknown)
- loan: has personal loan (yes, no, unknown)

Current Campaign Contact Information
- contact: type of communication (cellular, telephone)
- month: last contact month (jan–dec)
- day_of_week: last contact day (mon–fri)
- duration: last contact duration in seconds (note: should be excluded for realistic modeling since it’s only known after the call)

Previous Campaign Information
- campaign: number of contacts performed during this campaign
- pdays: number of days since last contact from a previous campaign (999 = not previously contacted)
- previous: number of contacts before this campaign
- poutcome: outcome of the previous campaign (failure, nonexistent, success)

Economic Indicators
- emp.var.rate: employment variation rate (quarterly indicator)
- cons.price.idx: consumer price index (monthly indicator)
- cons.conf.idx: consumer confidence index (monthly indicator)
- euribor3m: euribor 3 month rate (daily indicator)
- nr.employed: number of employees (quarterly indicator)

## Data Cleaning/Preprocessing
### Data Cleaning 
- Checked for duplicate rows and dropped them (41188 → 41176 records).
- No true NaN values were found in the dataset.
- -Converted the target column y from strings ("yes"/"no") to numeric (1/0).

### Preprocessing/Feature Engineering
#### Categorical Features
- Used One-Hot Encoding to convert categorical variables (job, marital, education, etc.) into binary dummy variables.
- For binary categorical features (housing, loan, default), applied drop-first encoding to avoid multicollinearity.

#### Numerical Features
- Standardized continuous numeric features (e.g., age, emp.var.rate, euribor3m) using StandardScaler to ensure models like Logistic Regression and SVM handle them correctly.

#### Feature Selection Considerations
- Dropped duration for the “realistic” model, since it is only known after a call is completed and would cause data leakage.
- Kept features from all major categories: client demographics, campaign attributes, and economic indicators.

#### Train/Test Split
- Split data into 80% training and 20% testing.
  
### EDA and Correlation Analysis
#### 1. Target Distribution
- The target variable y is highly imbalanced:
  - No: ~88%
  - Yes: ~12%
This imbalance highlights the need to use metrics beyond accuracy (e.g., Recall, F1, PR AUC).

#### 2. Numerical Features
- Age: Most clients are between 30–60 years old. Conversions are slightly higher among middle-aged and retired clients.
- Duration: Longer calls strongly correlate with positive outcomes, but this feature was excluded for realistic modeling.
- Campaign: Heavily right-skewed. More calls usually lower the chance of conversion → customer fatigue.
- Pdays: A spike at 999 (never contacted before). Smaller values (recent contact) are associated with higher conversions.
- Previous: Most values are 0. Nonzero indicates prior engagement and slightly higher conversion.
  <img src="images/price vs.transmission.png" width="600"/>

#### 3. Categorical Features
- Job: Largest groups are admin., blue-collar, technician. Students, retired, and management have higher conversion rates.
- Marital Status: Married is the largest group; single clients convert at slightly higher rates.
- Education: Higher education shows somewhat higher acceptance rates.
- Contact Method: Cellular contacts dominate volume and convert better than telephone.
- Month: Seasonal effects — higher campaign activity in May, but better conversion rates in March, October, and December.
- Day of Week: Fairly uniform distribution; little impact on conversion.
<img src="images/price vs.type.png" width="600"/>

---

## Model Performance Summary (with Cross-Validation):
- Each model was evaluated using a combination of 5-fold cross-validation on the training set and hold-out testing on unseen data. Here's what the performance metrics reveal:

<img src="images/table.png" width="850"/>

### Key Observations:


**Conclusion: .**


## Key Features Impacting Used Car Price


### Top Predictive Features effect on price: 


## Findings

### Business Understanding
- The Portuguese bank’s marketing campaigns generated large volumes of calls but yielded few successful term deposit subscriptions. The core business problem is inefficient targeting: agents often called clients unlikely to convert, leading to high costs, wasted effort, and customer frustration. The business need is to identify and prioritize clients who are most likely to subscribe, improving campaign efficiency and ROI.

### Actionable Insights (Nontechnical Language)

- Focus on warm leads: Clients with prior contact or positive outcomes are more likely to subscribe.
- Set call limits: Multiple calls in the same campaign reduce conversion likelihood — consider limiting repeated contacts.
- Optimize timing: Certain months (March, October, December) showed higher success rates — align resources accordingly.
- Channel strategy: Cellular contact consistently outperformed telephone outreach — prioritize mobile campaigns.

### Key Findings 
- Baseline: High accuracy but useless (never predicts positives).
- Logistic Regression: Best trade-off → strong recall, balanced AUCs, interpretable, efficient.
- SVC: Competitive with Logistic Regression but impractically slow for this dataset.
- KNN & Decision Tree: Misleadingly high accuracy but poor recall → miss most positive cases.


## Next Steps & Recommendations
- Adopt Logistic Regression as the preferred model: fast, interpretable, and strong at identifying subscribers.
- Explore ensemble methods (Random Forest, Gradient Boosting) for possible recall improvements.
- Operationalize model insights by limiting repeated contacts, focusing on high-potential clients, and adjusting staffing/resources to match seasonal peaks
- Build two versions of the model:
    -  Pre-call model (excludes duration) for realistic targeting.
    -  Post-call model (includes duration) for campaign analysis and benchmarking.




