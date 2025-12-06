import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from pytrends.request import TrendReq
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
    from xgboost import XGBClassifier
    import os
except ImportError as e:
    st.error(f"Missing required package: {e}")
    st.stop()

# Page configuration
st.set_page_config(page_title="Movie Trend Predictor", page_icon="🎬", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem;}
    .subtitle {font-size: 1.5rem; text-align: center; margin-bottom: 2rem; font-weight: 400;}
    .prediction-box {padding: 1.5rem; border-radius: 10px; margin: 1rem 0;}
    .emerging {background-color: #d4edda; border: 2px solid #28a745;}
    .stable {background-color: #fff3cd; border: 2px solid #ffc107;}
    .input-label {font-size: 1.3rem; font-weight: 500;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def fetch_trends_data(keyword: str, timeframe: str = "today 5-y", geo: str = "US"):
    """Fetch Google Trends data for a given keyword"""
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload([keyword], timeframe=timeframe, geo=geo, cat=23, gprop='')
        df = pytrends.interest_over_time()
        
        if df is None or df.empty:
            return None
        
        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
        
        series = df[keyword].astype(float)
        series.index = pd.to_datetime(series.index)
        return series
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def compute_features_from_series(series: pd.Series) -> dict:
    """Compute machine learning features from a time series"""
    s = series.astype(float).dropna().sort_index()
    if len(s) == 0:
        raise ValueError("Empty time series")
    
    feats = {}
    feats["level_last"] = float(s.iloc[-1])
    mean_w = float(s.mean())
    feats["mean_w"] = mean_w
    
    x = np.arange(len(s))
    slope, _ = np.polyfit(x, s.values, 1)
    feats["slope_w"] = float(slope)
    
    std_w = float(s.std()) if len(s) > 1 else 0.0
    feats["z_last"] = (feats["level_last"] - mean_w) / std_w if std_w > 0 else 0.0
    feats["lift_vs_mean_w"] = feats["level_last"] / (mean_w + 1e-9)
    feats["momentum_1"] = float(s.iloc[-1] - s.iloc[-2]) if len(s) >= 2 else 0.0
    feats["momentum_7"] = float(s.iloc[-1] - s.iloc[-8]) if len(s) >= 8 else 0.0
    feats["coefvar_w"] = std_w / (mean_w + 1e-9)
    
    if hasattr(s.index, 'days'):
        peak_idx = s.idxmax()
        feats["days_since_peak"] = float((s.index.max() - peak_idx).days)
    else:
        peak_pos = int(s.values.argmax())
        feats["days_since_peak"] = float(len(s) - 1 - peak_pos)
    
    feats["peak"] = float(s.max())
    return feats

def build_single_row_features(series: pd.Series, feature_cols, train_df):
    """Build feature DataFrame for a single keyword's time series"""
    if train_df is None or feature_cols is None:
        return None
        
    base = train_df[feature_cols].mean()
    row = base.to_dict()
    
    try:
        engineered = compute_features_from_series(series)
        for k, v in engineered.items():
            if k in row:
                row[k] = v
    except Exception as e:
        st.warning(f"Feature computation warning: {e}")
    
    return pd.DataFrame([row])

@st.cache_resource
def train_model_from_data(data_path: str = "labeled_trends.csv"):
    """Train XGBoost model from labeled trends dataset"""
    if not os.path.exists(data_path):
        return None, None, None, None, None, None
    
    try:
        df = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        EXCLUDE_COLS = ['date', 'keyword', 'target', 'growth_label_metric', 'future_mean_7d', 'future_max_7d', 'future_sum_7d']
        feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if df[col].notna().sum() > 0]
        
        test_cutoff_date = df['date'].max() - pd.DateOffset(months=6)
        train_mask = df['date'] < test_cutoff_date
        test_mask = df['date'] >= test_cutoff_date
        
        X_train_raw = df[train_mask][feature_cols]
        y_train = df[train_mask]['target']
        X_test_raw = df[test_mask][feature_cols]
        y_test = df[test_mask]['target']
        train_df = df[train_mask].copy()
        
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=feature_cols, index=X_train_raw.index)
        X_test_imputed = pd.DataFrame(imputer.transform(X_test_raw), columns=feature_cols, index=X_test_raw.index)
        
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=feature_cols, index=X_train_imputed.index)
        X_test = pd.DataFrame(scaler.transform(X_test_imputed), columns=feature_cols, index=X_test_imputed.index)
        
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        
        XGB_PARAMS = {
            'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'eval_metric': 'logloss',
            'scale_pos_weight': weights[1] / weights[0]
        }
        
        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.35).astype(int)
        metrics = {
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }
        
        model_data = {'model_name': 'XGBoost', 'metrics': metrics, 'threshold': 0.35}
        return model, scaler, imputer, feature_cols, train_df, model_data
        
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None, None, None, None, None

def predict_emerging(features_df, threshold, model, scaler, imputer, feature_cols):
    """Predict if a keyword is emerging based on its features"""
    if features_df is None or features_df.empty:
        return 0.0, "INSUFFICIENT_DATA"
    
    if model is None or imputer is None or feature_cols is None:
        st.error("Model not available. Please ensure the training data file exists.")
        return 0.0, "MODEL_UNAVAILABLE"
    
    try:
        missing_cols = set(feature_cols) - set(features_df.columns)
        for col in missing_cols:
            features_df[col] = np.nan
        
        X = features_df[feature_cols].copy()
        X_imputed = imputer.transform(X)
        X_scaled = scaler.transform(X_imputed) if scaler is not None else X_imputed
        
        prob = float(model.predict_proba(X_scaled)[0, 1])
        label = "EMERGING" if prob >= threshold else "STABLE"
        return prob, label
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.0, "PREDICTION_ERROR"

def main():
    st.markdown('<div class="main-header">🎬 AI-Powered Movie Trend Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Predict whether a movie-related keyword is emerging or stable based on Google Trends data</div>', unsafe_allow_html=True)
    
    with st.spinner("Loading model..."):
        model, scaler, imputer, feature_cols, train_df, model_data = train_model_from_data()
    
    default_threshold = model_data.get('threshold', 0.35) if model_data else 0.35
    
    with st.sidebar:
        st.header("⚙️ Settings")
        timeframe = st.selectbox("Time Range", ["today 5-y", "today 3-y", "today 12-m", "today 3-m"], index=0)
        threshold = st.slider("Prediction Threshold", 0.1, 0.9, default_threshold, 0.05,
                            help="Lower threshold = more sensitive (catches more trends but more false alarms)")
        st.markdown("---")
        st.markdown("### 📊 About")
        st.markdown("""
        This app uses Google Trends data to predict whether a movie-related keyword is:
        - **EMERGING**: Likely to grow in popularity
        - **STABLE**: Maintaining current interest levels
        
        Enter keywords below to see predictions and visualizations!
        """)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="input-label">🔍 Enter a movie-related keyword:</div>', unsafe_allow_html=True)
        keyword = st.text_input("Keyword", placeholder="e.g., 'Stranger Things Season 5'", key="keyword_input", label_visibility="collapsed")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button("🚀 Analyze Trend", type="primary", use_container_width=True)
    
    if analyze_button and keyword:
        with st.spinner(f"Fetching Google Trends data for '{keyword}'..."):
            series = fetch_trends_data(keyword, timeframe=timeframe, geo="US")
        
        if series is None or len(series) == 0:
            st.error(f"❌ No data found for '{keyword}'")
        else:
            with st.spinner("Analyzing trends..."):
                features_df = build_single_row_features(series, feature_cols, train_df)
                prob, label = predict_emerging(features_df, threshold, model, scaler, imputer, feature_cols)
            
            st.markdown("---")
            
            result_class = "emerging" if label == "EMERGING" else "stable"
            result_emoji = "📈" if label == "EMERGING" else "📊"
            
            st.markdown(f"""
            <div class="prediction-box {result_class}">
                <h2 style="color: black;">{result_emoji} Prediction: {label}</h2>
                <h3 style="color: black;">Confidence: {prob:.1%}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Interest", f"{series.iloc[-1]:.0f}", 
                         delta=f"{series.iloc[-1] - series.iloc[-7] if len(series) >= 7 else 0:.1f} vs 7d ago")
            with col2:
                recent_avg = series.iloc[-28:].mean() if len(series) >= 28 else series.mean()
                st.metric("28-Day Average", f"{recent_avg:.0f}")
            with col3:
                st.metric("Peak Interest", f"{series.max():.0f}")
            with col4:
                if features_df is not None and not features_df.empty:
                    slope = features_df.iloc[0].get('slope_w', np.nan)
                    st.metric("Trend Slope", f"{slope:.3f}" if not np.isnan(slope) else "N/A")
            
            st.markdown("---")
            
            st.subheader("📈 Trend Analysis")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines+markers', 
                                    name='Interest Over Time', line=dict(color='#1f77b4', width=2), 
                                    marker=dict(size=4)))
            
            if len(series) >= 28:
                ma28 = series.rolling(28).mean()
                fig.add_trace(go.Scatter(x=ma28.index, y=ma28.values, mode='lines', 
                                        name='28-Day MA', line=dict(color='orange', width=2, dash='dash')))
            
            fig.update_layout(
                title=f"Google Trends Interest: '{keyword}'", 
                xaxis_title="Date", 
                yaxis_title="Interest Score",
                hovermode='x unified', 
                height=400, 
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Monthly Average Interest")
                monthly_data = series.resample('M').mean()
                
                fig_monthly = go.Figure()
                fig_monthly.add_trace(go.Bar(
                    x=monthly_data.index,
                    y=monthly_data.values,
                    marker_color='steelblue',
                    name='Monthly Average'
                ))
                fig_monthly.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Average Interest",
                    height=350,
                    template="plotly_white"
                )
                st.plotly_chart(fig_monthly, use_container_width=True)
            
            with col2:
                st.subheader("📉 Interest Distribution")
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=series.values,
                    nbinsx=30,
                    marker_color='lightcoral',
                    name='Distribution'
                ))
                fig_hist.update_layout(
                    xaxis_title="Interest Score",
                    yaxis_title="Frequency",
                    height=350,
                    template="plotly_white",
                    showlegend=False
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            st.subheader("📈 Volatility & Momentum Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                rolling_std = series.rolling(28).std()
                fig_volatility = go.Figure()
                fig_volatility.add_trace(go.Scatter(
                    x=rolling_std.index,
                    y=rolling_std.values,
                    mode='lines',
                    fill='tozeroy',
                    name='28-Day Volatility',
                    line=dict(color='purple', width=2)
                ))
                fig_volatility.update_layout(
                    title="Interest Volatility (28-Day Std Dev)",
                    xaxis_title="Date",
                    yaxis_title="Volatility",
                    height=350,
                    template="plotly_white"
                )
                st.plotly_chart(fig_volatility, use_container_width=True)
            
            with col2:
                momentum = series.diff(7)
                fig_momentum = go.Figure()
                colors = ['green' if x > 0 else 'red' for x in momentum.values]
                fig_momentum.add_trace(go.Bar(
                    x=momentum.index,
                    y=momentum.values,
                    marker_color=colors,
                    name='7-Day Momentum'
                ))
                fig_momentum.update_layout(
                    title="7-Day Momentum (Change)",
                    xaxis_title="Date",
                    yaxis_title="Change in Interest",
                    height=350,
                    template="plotly_white",
                    showlegend=False
                )
                st.plotly_chart(fig_momentum, use_container_width=True)
            
            st.subheader("🔄 Recent vs Historical Performance")
            
            recent_cutoff = series.index.max() - pd.DateOffset(months=3)
            recent_data = series[series.index >= recent_cutoff]
            historical_data = series[series.index < recent_cutoff]
            
            comparison_data = {
                'Period': ['Historical', 'Recent (Last 3 Months)'],
                'Average Interest': [historical_data.mean(), recent_data.mean()],
                'Peak Interest': [historical_data.max(), recent_data.max()],
                'Volatility (Std)': [historical_data.std(), recent_data.std()]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            
            fig_comparison = go.Figure()
            fig_comparison.add_trace(go.Bar(
                name='Average Interest',
                x=comparison_df['Period'],
                y=comparison_df['Average Interest'],
                marker_color='lightblue'
            ))
            fig_comparison.add_trace(go.Bar(
                name='Peak Interest',
                x=comparison_df['Period'],
                y=comparison_df['Peak Interest'],
                marker_color='darkblue'
            ))
            fig_comparison.update_layout(
                xaxis_title="Time Period",
                yaxis_title="Interest Score",
                barmode='group',
                height=350,
                template="plotly_white"
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            if features_df is not None and not features_df.empty:
                st.subheader("🔍 Key Predictive Indicators")
                row = features_df.iloc[0]
                
                key_features = {
                    'Trend Slope': row.get('slope_w', 0),
                    'Lift vs Mean': row.get('lift_vs_mean_w', 0) * 100,
                    '7-Day Momentum': row.get('momentum_7', 0),
                    'Z-Score': row.get('z_last', 0),
                    'Trend Strength': row.get('trend_strength', 0) * 100
                }
                key_features = {k: v for k, v in key_features.items() if not np.isnan(v)}
                
                if key_features:
                    fig_features = go.Figure()
                    colors = ['green' if v > 0 else 'red' for v in key_features.values()]
                    fig_features.add_trace(go.Bar(
                        x=list(key_features.keys()), 
                        y=list(key_features.values()),
                        marker_color=colors, 
                        text=[f"{v:.2f}" for v in key_features.values()], 
                        textposition='outside'
                    ))
                    fig_features.update_layout(
                        title="Key Trend Indicators (Positive = Emerging Signal)",
                        xaxis_title="Indicator", 
                        yaxis_title="Value", 
                        height=600, 
                        template="plotly_white", 
                        showlegend=False
                    )
                    st.plotly_chart(fig_features, use_container_width=True)
            
            st.markdown("---")
            st.subheader("💡 Interpretation")
            
            if label == "EMERGING":
                st.success(f"""
                **{keyword}** shows signs of being an **EMERGING** trend!
                
                - The keyword has a **{prob:.1%}** probability of experiencing growth
                - This suggests increasing interest that may continue to rise
                - Consider monitoring this trend closely for marketing opportunities
                """)
            else:
                st.info(f"""
                **{keyword}** appears to be **STABLE** at the moment.
                
                - The keyword has a **{prob:.1%}** probability of being emerging
                - Current interest levels are relatively consistent
                - This doesn't mean it won't grow, but current indicators suggest stability
                """)
    
    elif not keyword and analyze_button:
        st.warning("⚠️ Please enter a keyword to analyze.")

if __name__ == "__main__":
    main()