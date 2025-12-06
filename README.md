# Google Trends Entertainment Predictor
### AI-Powered Movie Trend Spotting System
---

### 👥 **Team Members**

| Name             | University | GitHub | Contribution                                                             |
|------------------|------------|---------------|--------------------------------------------------------------------------|
| Vidula Alla      | UW-Madison | [Vidula GitHub](https://github.com/vidulaalla26) | Data exploration, visualization, overall project coordination|
| Fatima Atif      | UT San Antonio| [Fatima GitHub](https://github.com/fatimaagit)      | Data collection, exploratory data analysis (EDA), dataset documentation  |
| Angela Martinez  | UT El Paso | [Angela GitHub](https://github.com/) | Data preprocessing, feature engineering, data validation                 |
| Namyuktha Prakash| UT Dallas  | [Namyuktha GitHub](https://github.com/namyuktha-prakash)| Model selection, hyperparameter tuning, model training and optimization  |
| Neeju Singh      | Metropolitan State | [ Neeju GitHub](https://github.com/NEEJUSINGH)    | Model evaluation, performance analysis, results interpretation           |
| Aadhi Sivakumar  | UT Dallas  | [Aadhi GitHub](https://github.com/aadhi-sivakumar)   | Data Preprocessing,  Exploratory Data Analysis (EDA), Streamlit Website           |
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

### Prerequisites
- Python 3.8+
- Google Colab account (for running notebooks)
- Code editor (VS Code, PyCharm, etc.) for Streamlit app
- Internet connection for PyTrends API access

---

### **Option 1: Full Pipeline (Complete Process)**

Run `Google_Trends_Movie_Predictor__Final.ipynb` to execute the entire pipeline from data collection to model training.

#### Steps:

1. **Download the notebook**
   - Download `Google_Trends_Movie_Predictor__Final.ipynb` to your computer

2. **Open Google Colab**
   - Visit [https://colab.research.google.com/](https://colab.research.google.com/)

3. **Upload the notebook**
   - In Colab, click on **Upload**
   - Select the `Google_Trends_Movie_Predictor__Final.ipynb` file from your computer

4. **Change the runtime to T4 GPU**
   - Click **Runtime** → **Change runtime type** → Select **T4 GPU** from the dropdown menu
   - Click **Save**

5. **Run the cells**
   
   You can either:
   - Click the **Run (▶️)** button on the left of each cell to run them one by one, OR
   - Click **Runtime** → **Run all** from the menu bar to execute all cells at once

6. **Execute sequentially**
   - ⚠️ Make sure you run all cells sequentially from top to bottom
   - ⚠️ Do not skip any cells, since later parts of the notebook may depend on earlier ones
   - The notebook will generate `labeled_trends.csv` which is used for model training

---

### **Option 2: Quick Start (Using Pre-labeled Data)**

If you already have `labeled_trends.csv`, you can skip the data collection process and go straight to model training.

#### Steps:

1. **Download and upload the notebook**
   - Download `live_prediction_model.ipynb` to your computer
   - Upload to Google Colab (same process as Option 1)

2. **Run the notebook**
   - Execute all cells sequentially
   - This notebook skips the data collection process and uses the pre-labeled data

---

### **Running the Streamlit Dashboard**

To launch the interactive web application:

#### Steps:

1. **Clone the repository**
```bash
git clone
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify the dataset**
   - ⚠️ Ensure `labeled_trends.csv` is present in your project directory
   - If not, run one of the notebooks above to generate it

4. **Run the Streamlit app**
```bash
streamlit run streamlit_app.py
```

5. **Access the dashboard**
   - The app will open automatically in your default browser
   - If not, navigate to the URL shown in the terminal (typically `http://localhost:8501`)

---
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

### Limitations
- **Loading in PyTrends data without 429 rate limit error**
  - API rate limiting was a constant challenge requiring exponential backoff strategies
- **Preparing our data for modeling and evaluation**
  - Ensuring that the data was cleaned properly & handling missing values
  - Extensive preprocessing required to maintain data quality
- **Deciding whether to split our data using test train split in scikit-learn or do a time-based split**
  - Tried test train split → bad evaluation metric scores
  - Time-based split was necessary to prevent data leakage and simulate real-world deployment

### Future Enhancements

**Data Expansion**
- Train on Global Data
  - Reach a wider range of users
  - Understand regional differences in trend patterns
- Add more temporal features or external data sources (social media, box office data)
  - Incorporate Twitter/X sentiment analysis
  - Integrate actual box office performance data
  - Include streaming platform viewership metrics
- Automate data pipeline with scheduled PyTrends fetches
  - Implement daily/weekly automated data collection
  - Build data versioning system for reproducibility

**Prediction Enhancements**
- Move from binary classification ("emerging" vs "stable") to multi-class or continuous trend forecasting
  - Add "declining" and "viral" categories
  - Predict magnitude of trend growth, not just direction
  - Forecast trend trajectory over time horizons (7-day, 30-day, 90-day)

**Dashboard Upgrades**
- Add interactive alerting system for new emerging trends
  - Email/SMS notifications when trends cross emergence threshold
  - Customizable alert criteria based on user preferences

---

## 🙏 **Acknowledgements**

We extend our heartfelt gratitude to:

- **Haziel Ayala**, our AI Studio Coach, for invaluable guidance throughout the project lifecycle
- **Sarah Tan** and **Hunter Saine**, our Google Challenge Advisors, for sharing industry insights and technical expertise
- **Google** and the **Break Through Tech AI Program** for providing this incredible learning opportunity
