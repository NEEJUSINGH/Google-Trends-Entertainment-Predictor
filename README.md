# Google Trends Entertainment Predictor
### AI-Powered Movie Trend Spotting System
---

### 👥 **Team Members**

| Name             | University | GitHub | Contribution                                                             |
|------------------|------------|---------------|--------------------------------------------------------------------------|
| Vidula Alla      | UW-Madison |  | Data exploration, visualization, overall project coordination|
| Fatima Atif      | UT San Antonio|      | Data collection, exploratory data analysis (EDA), dataset documentation  |
| Angela Martinez  | UT El Paso | @aminahassan  | Data preprocessing, feature engineering, data validation                 |
| Namyuktha Prakash| UT Dallas  | @pmehta       | Model selection, hyperparameter tuning, model training and optimization  |
| Neeju Singh      | Metropolitan State | @chrispark    | Model evaluation, performance analysis, results interpretation           |
| Aadhi Sivakumar  | UT Dallas  | @chrispark    | Model evaluation, performance analysis, results interpretation           |
| Alison Zou       | Vanderbilt | [Alison Github](https://github.com/azzou02)  | Model evaluation, performance analysis, results interpretation           |

### **AI Studio Coach:** Haziel Ayala  
### **Challenge Advisors:** Sarah Tan & Hunter Saine (Google Software Engineers)
---

## 🎯 **Project Highlights**

- Developed an AI-powered system that analyzes Google Trends data to automatically detect emerging topics in entertainment, specifically movies
- Achieved **86.4% accuracy** using XGBoost classifier (F1 Score: 0.456) in predicting whether movie-related keywords are emerging or stable
- Built an interactive **Streamlit dashboard** allowing end users to input movie keywords and receive real-time trend predictions
- Engineered **30-40 sophisticated features** including rolling window statistics, trend slopes, STL decomposition, and cohort-based metrics
- Processed **5 years of US Google Trends data** across movie genres, themes, tropes, and industry terms
- Implemented leakage-safe feature generation ensuring true real-world forecasting behavior

---


## 👩🏽‍💻 **Setup and Installation**

**Provide step-by-step instructions so someone else can run your code and reproduce your results. Depending on your setup, include:**

* How to clone the repository
* How to install dependencies
* How to set up the environment
* How to access the dataset(s)
* How to run the notebook or scripts

---

## 🏗️ **Project Overview**

This project was developed as part of the **Break Through Tech AI Program** in collaboration with **Google**. Our team was challenged to leverage Google's vast data ecosystem to create a practical machine learning solution.

### Business Problem
Entertainment studios face critical decisions about:
- When to greenlight projects aligned with future audience demand
- Optimal timing for marketing campaigns and content releases
- Which emerging trends represent genuine market opportunities vs. temporary noise

### Solution
We built an AI system that:
1. Continuously monitors Google Trends for movie-related keywords
2. Analyzes temporal patterns using sophisticated feature engineering
3. Classifies trends as "EMERGING" (likely to grow) or "STABLE" (maintaining current interest)
4. Provides actionable insights through an interactive dashboard

### Business Impact
- **Smarter Content Investment:** Enter markets ahead of trend saturation, gaining first-mover advantage
- **Risk Reduction:** Greenlight projects backed by predictive analytics rather than intuition
- **Revenue Forecast Accuracy:** Better prediction of audience interest translates to improved financial planning
- **Marketing Optimization:** Identify optimal windows for trailers, social campaigns, and PR pushes

---

## 📊 **Data Exploration**

### Dataset Overview
- **Source:** Google Trends API (via PyTrends library)
- **Scope:** 5 years of US-based search interest data
- **Size:** Thousands of movie-related keywords tracked daily
- **Keywords Include:**
  - Movie genres (action, horror, romance, sci-fi)
  - Themes and tropes (superhero, time travel, heist)
  - Industry terms (box office, premiere, streaming)
  - Related keywords discovered through cohort enrichment

### Data Collection Process
1. **Seed Keywords:** Started with curated list of movie genres, themes, and tropes
2. **Cohort Enrichment:** Used PyTrends' related queries feature to expand keyword universe
3. **API Management:** Implemented exponential backoff to handle rate limiting
4. **Batch Processing:** Queried data in batches to comply with API limitations

### Key Preprocessing Steps
- **Handling Missing Values:** Interpolation and forward-fill strategies
- **Normalization:** Google Trends provides 0-100 scaled values
- **Temporal Alignment:** Ensured consistent daily granularity across all keywords
- **Leakage Prevention:** Features only use data up to time t, labels use future data

### EDA Insights
- **High Correlation Patterns:** Related movie keywords (e.g., "Stranger Things" and "Stranger Things Season 5") show strong correlation
- **Seasonality:** Holiday-themed content shows clear annual patterns
- **Volatility:** Franchise releases create sharp, predictable spikes
- **Trend Lifecycle:** Most trends follow: emergence → peak → plateau → decline

### Visualizations
- **Time Series Plots:** Google Trends over 5 years for sample keywords
- **Correlation Heatmap:** Relationships between different movie categories
- **Feature Distribution:** Statistical properties of engineered features

---

## 🧠 **Model Development**

### Feature Engineering (30-40 Features)
Our leakage-safe feature generation ensures predictions use only historical data:

**Rolling Window Statistics (28-day)**
- Mean, median, standard deviation
- Captures recent trend momentum

**Reference Window Baseline (90-day)**
- Establishes longer-term baseline
- Enables detection of deviation from normal

**Trend Metrics**
- Linear regression slope over recent period
- Z-score normalization
- Days since peak value

**STL Decomposition**
- Trend strength
- Seasonality strength
- Residual component analysis

**Cohort-Based Features**
- Median lift compared to related keywords
- Correlation with keyword cohort
- Relative performance metrics

### Models Trained
1. **XGBoost** ⭐ (Best Performer)
2. LightGBM
3. Random Forest
4. Gradient Boosting Classifier
5. Logistic Regression (baseline)

### Training Strategy
- **Time-Based Split:** Train on historical data, test on most recent 6 months
  - Simulates real-world deployment scenario
  - Prevents data leakage
- **Class Weights:** Applied to handle imbalanced classes
- **Preprocessing Pipeline:** Imputation → Scaling → SMOTE (optional)

### Hyperparameter Tuning
- Pre-configured optimal parameters for each model
- Focused on maximizing F1 score (balanced precision/recall)

### Why Multiple Models?
- Compare tree-based ensembles vs. linear baselines
- Time-series classification benefits from nonlinear pattern detection
- Ensemble approaches can be explored for further improvement

---

## 📈 **Results & Key Findings**

### Model Performance Comparison

| Model                    | Accuracy | F1 Score | Training Time |
|--------------------------|----------|----------|---------------|
| **XGBoost** ⭐           | **0.864** | **0.456** | Fast          |
| Gradient Boosting        | 0.855    | 0.441    | Medium        |
| Ensemble (XGB+LGBM)      | 0.852    | 0.440    | Medium        |
| LightGBM                 | 0.848    | 0.432    | Very Fast     |
| Random Forest            | 0.838    | 0.401    | Slow          |
| Logistic Regression      | 0.467    | 0.216    | Very Fast     |

### XGBoost Confusion Matrix
- **True Positives:** 1239 (Correctly predicted emerging trends)
- **True Negatives:** 156 (Correctly predicted stable trends)
- **False Positives:** 52 (Predicted emerging, but was stable)
- **False Negatives:** 87 (Missed emerging trend - worst case!)

---

## 🚀 **Next Steps**

**You might consider addressing the following (as applicable):**

* What are some of the limitations of your model?
* What would you do differently with more time/resources?
* What additional datasets or techniques would you explore?

---

## 📄 **References** (Optional but encouraged)

Cite relevant papers, articles, or resources that supported your project.

---

## 🙏 **Acknowledgements**

We extend our heartfelt gratitude to:

- **Haziel Ayala**, our AI Studio Coach, for invaluable guidance throughout the project lifecycle
- **Sarah Tan** and **Hunter Saine**, our Google Challenge Advisors, for sharing industry insights and technical expertise
