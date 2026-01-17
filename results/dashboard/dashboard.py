"""
Credit Risk Assessment Dashboard
================================
A professional Streamlit dashboard for credit scoring model visualization.
Designed for Home Credit Default Risk assessment.

Author: Amir Iqbal
"""
import streamlit as st
import pandas as pd
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import numpy as np

# Set page config
st.set_page_config(
    page_title="Credit Risk Assessment | Home Credit",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional banking look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #1a365d;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        text-align: center;
        color: white;
    }
    .risk-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .risk-medium {
        background: linear-gradient(135deg, #F2994A 0%, #F2C94C 100%);
    }
    .risk-high {
        background: linear-gradient(135deg, #CB356B 0%, #BD3F32 100%);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 12px 30px;
        font-weight: bold;
        font-size: 16px;
        transition: transform 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 5px solid #2980b9;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .stat-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "results" / "model"

# Feature Name Mapping for business users
FEATURE_MAPPING = {
    "EXT_SOURCE_1": "External Credit Score 1",
    "EXT_SOURCE_2": "External Credit Score 2", 
    "EXT_SOURCE_3": "External Credit Score 3",
    "EXT_SOURCE_MEAN": "Average External Score",
    "EXT_SOURCE_PROD": "Combined External Score",
    "AMT_CREDIT": "Loan Amount",
    "AMT_ANNUITY": "Monthly Payment",
    "AMT_INCOME_TOTAL": "Annual Income",
    "AMT_GOODS_PRICE": "Goods Price",
    "CREDIT_INCOME_RATIO": "Loan-to-Income Ratio",
    "ANNUITY_INCOME_RATIO": "Payment-to-Income Ratio",
    "CREDIT_GOODS_RATIO": "Loan-to-Goods Ratio",
    "AGE_YEARS": "Age (Years)",
    "EMPLOYMENT_YEARS": "Employment Duration (Years)",
    "INCOME_PER_FAMILY": "Income per Family Member",
    "DAYS_BIRTH": "Age",
    "DAYS_EMPLOYED": "Employment Duration",
    "NAME_CONTRACT_TYPE": "Contract Type",
    "CODE_GENDER": "Gender",
    "FLAG_OWN_CAR": "Owns Vehicle",
    "FLAG_OWN_REALTY": "Owns Property",
    "CNT_CHILDREN": "Number of Children",
    "CNT_FAM_MEMBERS": "Family Size",
    "NAME_EDUCATION_TYPE": "Education Level",
    "NAME_FAMILY_STATUS": "Marital Status",
    "NAME_HOUSING_TYPE": "Housing Type",
    "ORGANIZATION_TYPE": "Employer Type",
    "bureau_row_count": "Bureau Records",
    "prev_row_count": "Previous Applications",
    "DOCUMENT_COUNT": "Documents Provided",
    "CONTACT_SCORE": "Contact Availability",
}

def format_feature_name(name):
    """Convert technical feature name to business-friendly name."""
    if name in FEATURE_MAPPING:
        return FEATURE_MAPPING[name]
    return name.replace('_', ' ').title()[:35]

def format_currency(value):
    """Format number as currency."""
    if abs(value) >= 1e6:
        return f"${value/1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"${value/1e3:.0f}K"
    return f"${value:.0f}"

@st.cache_resource
def load_model():
    """Load the trained pipeline."""
    model_path = MODEL_DIR / "baseline_logreg.joblib"
    return joblib.load(model_path)

@st.cache_data
def load_data():
    """Load train features for context."""
    df = pd.read_csv(DATA_DIR / "application_train_features.csv")
    df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(int)
    df.set_index('SK_ID_CURR', inplace=True)
    return df

@st.cache_data
def load_test_data():
    """Load test features for context."""
    df = pd.read_csv(DATA_DIR / "application_test_features.csv")
    df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(int)
    df.set_index('SK_ID_CURR', inplace=True)
    return df

def align_features(df, pipeline):
    """Align features with model expectations."""
    scaler = pipeline.named_steps['standardscaler']
    
    X = df.drop(columns=["TARGET"], errors='ignore')
    cat_cols = X.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)

    if hasattr(pipeline, "feature_names_in_"):
        expected_features = pipeline.feature_names_in_
    elif hasattr(scaler, "feature_names_in_"):
        expected_features = scaler.feature_names_in_
    else:
        expected_features = X.columns

    X = X.reindex(columns=expected_features, fill_value=0)
    X = X.replace([float("inf"), float("-inf")], np.nan).fillna(0)
    return X, expected_features

def get_risk_category(prob):
    """Categorize risk level."""
    if prob < 0.15:
        return "LOW", "#38ef7d", "✅"
    elif prob < 0.35:
        return "MODERATE", "#F2C94C", "⚠️"
    elif prob < 0.55:
        return "ELEVATED", "#F2994A", "⚠️"
    else:
        return "HIGH", "#BD3F32", "🚨"

def create_gauge_chart(prob):
    """Create a professional gauge chart for risk score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Default Risk Score", 'font': {'size': 20, 'color': '#1a365d'}},
        number={'suffix': '%', 'font': {'size': 40, 'color': '#1a365d'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#1a365d"},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#1a365d",
            'steps': [
                {'range': [0, 15], 'color': '#38ef7d'},
                {'range': [15, 35], 'color': '#F2C94C'},
                {'range': [35, 55], 'color': '#F2994A'},
                {'range': [55, 100], 'color': '#CB356B'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': prob * 100
            }
        }
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Segoe UI'}
    )
    return fig

# Initialize session state for persisting analysis results
if 'analyzed_client_id' not in st.session_state:
    st.session_state.analyzed_client_id = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Load resources
with st.spinner('🏦 Initializing Credit Assessment System...'):
    try:
        pipeline = load_model()
        scaler = pipeline.named_steps['standardscaler']
        classifier = pipeline.named_steps['logisticregression']
        
        # Load both train and test data
        raw_df_train = load_data()
        raw_df_test = load_test_data()
        
        X_train, feature_names = align_features(raw_df_train, pipeline)
        X_test, _ = align_features(raw_df_test, pipeline)
        
        # Combine for full access
        X = pd.concat([X_train, X_test])
        raw_df = pd.concat([raw_df_train, raw_df_test])
        
        # Track which IDs are from test set
        test_ids = set(X_test.index)
        
        # Initialize SHAP Explainer
        background_sample = X_train.sample(min(100, len(X_train)), random_state=42)
        background_scaled = scaler.transform(background_sample)
        background_scaled_df = pd.DataFrame(background_scaled, columns=X.columns)
        explainer = shap.LinearExplainer(classifier, background_scaled_df)
        
        model_loaded = True
    except Exception as e:
        st.error(f"⚠️ System Error: {e}")
        model_loaded = False

# Sidebar
st.sidebar.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1 style='color: #1a365d;'>🏦</h1>
    <h2 style='color: #1a365d; margin: 0;'>Home Credit</h2>
    <p style='color: #666;'>Risk Assessment System</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Model Stats
st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.markdown("""
<div class='info-box'>
    <strong>AUC Score:</strong> 0.76<br>
    <strong>Model:</strong> Logistic Regression<br>
    <strong>Features:</strong> 290
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Client Lookup")

# Sample client IDs for demo
sample_ids = [100002, 100003, 100004, 100006, 100007]
sample_ids = [sid for sid in sample_ids if sid in X.index]

client_id = st.sidebar.number_input(
    "Enter Client ID", 
    value=int(X.index[0]), 
    help="Enter the unique SK_ID_CURR"
)

if sample_ids:
    st.sidebar.markdown("**Quick Access (Sample IDs):**")
    cols = st.sidebar.columns(3)
    for i, sid in enumerate(sample_ids[:6]):
        if cols[i % 3].button(str(sid), key=f"sample_{sid}"):
            st.session_state.analyzed_client_id = sid

analyze_btn = st.sidebar.button("🔎 Analyze Risk Profile", width='stretch')

# Store analysis when button is clicked
if analyze_btn and model_loaded:
    st.session_state.analyzed_client_id = client_id

# Clear button to reset
if st.session_state.analyzed_client_id is not None:
    if st.sidebar.button("🔄 Clear Analysis", width='stretch'):
        st.session_state.analyzed_client_id = None
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; font-size: 12px; color: #666;'>
    <p>Built with Streamlit & SHAP</p>
    <p>© 2026 Credit Risk Analytics</p>
</div>
""", unsafe_allow_html=True)

# Main Content
st.markdown("""
<h1 style='text-align: center; color: #1a365d; margin-bottom: 0;'>
    Credit Risk Assessment Dashboard
</h1>
<p style='text-align: center; color: #666; font-size: 18px;'>
    AI-Powered Default Probability Analysis
</p>
""", unsafe_allow_html=True)

if model_loaded and st.session_state.analyzed_client_id is not None:
    # Use the stored client ID from session state
    analyzed_id = st.session_state.analyzed_client_id
    
    if analyzed_id not in X.index:
        st.error(f"❌ Client ID **{analyzed_id}** not found in database. Please verify the ID.")
    else:
        # Check if client is from test or train set
        is_test_client = analyzed_id in test_ids
        client_source = "Test Set" if is_test_client else "Training Set"
        source_badge = "🧪" if is_test_client else "📊"
        
        # Get client data
        client_data = X.loc[[analyzed_id]]
        client_raw = raw_df.loc[[analyzed_id]] if analyzed_id in raw_df.index else None
        
        # Predict
        client_scaled = scaler.transform(client_data)
        prob = classifier.predict_proba(client_scaled)[0][1]
        risk_level, risk_color, risk_icon = get_risk_category(prob)
        
        st.markdown("---")
        
        # Header with client info
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a365d 0%, #2d3748 100%); 
                    padding: 20px; border-radius: 15px; color: white; margin-bottom: 20px;'>
            <h2 style='margin: 0; color: white;'>{risk_icon} Client Assessment: #{analyzed_id}</h2>
            <p style='margin: 5px 0 0 0; opacity: 0.8;'>
                {source_badge} <strong>{client_source}</strong> | Comprehensive risk analysis based on 290 features
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.plotly_chart(create_gauge_chart(prob), width='stretch')
        
        with col2:
            st.markdown(f"""
            <div class='stat-card'>
                <h4 style='color: #666; margin: 0;'>Risk Level</h4>
                <h2 style='color: {risk_color}; margin: 10px 0;'>{risk_level}</h2>
                <p style='color: #999; margin: 0;'>Classification</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            if client_raw is not None and 'AMT_CREDIT' in client_raw.columns:
                loan_amt = client_raw['AMT_CREDIT'].values[0]
                st.markdown(f"""
                <div class='stat-card'>
                    <h4 style='color: #666; margin: 0;'>Loan Amount</h4>
                    <h2 style='color: #1a365d; margin: 10px 0;'>{format_currency(loan_amt)}</h2>
                    <p style='color: #999; margin: 0;'>Requested</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            if client_raw is not None and 'AMT_INCOME_TOTAL' in client_raw.columns:
                income = client_raw['AMT_INCOME_TOTAL'].values[0]
                st.markdown(f"""
                <div class='stat-card'>
                    <h4 style='color: #666; margin: 0;'>Annual Income</h4>
                    <h2 style='color: #1a365d; margin: 10px 0;'>{format_currency(income)}</h2>
                    <p style='color: #999; margin: 0;'>Declared</p>
                </div>
                """, unsafe_allow_html=True)

        # Decision recommendation
        st.markdown("---")
        if prob < 0.2:
            st.success(f"✅ **RECOMMENDATION: APPROVE** - This client shows strong creditworthiness with a {prob:.1%} default probability.")
        elif prob < 0.4:
            st.warning(f"⚠️ **RECOMMENDATION: REVIEW** - Moderate risk ({prob:.1%}). Additional verification recommended.")
        else:
            st.error(f"🚨 **RECOMMENDATION: DECLINE** - High risk profile ({prob:.1%}). Consider alternative products or collateral.")

        st.markdown("---")
        
        # SHAP Analysis
        st.markdown("## 🔍 Risk Factor Analysis")
        st.markdown("Understanding which factors contributed most to this assessment.")
        
        client_scaled_df = pd.DataFrame(client_scaled, columns=X.columns)
        shap_values = explainer.shap_values(client_scaled_df)
        
        sv = shap_values[0] if isinstance(shap_values, list) else shap_values
        if len(sv.shape) > 1:
            sv = sv[0]
            
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': sv
        })
        feature_importance['feature_name'] = feature_importance['feature'].apply(format_feature_name)
        feature_importance['abs_importance'] = feature_importance['importance'].abs()
        feature_importance = feature_importance.nlargest(15, 'abs_importance').sort_values('importance')
        
        # Split into positive and negative factors
        col_risk, col_safe = st.columns(2)
        
        with col_risk:
            st.markdown("### 🔴 Factors Increasing Risk")
            risk_factors = feature_importance[feature_importance['importance'] > 0].nlargest(7, 'importance')
            if len(risk_factors) > 0:
                fig_risk = px.bar(
                    risk_factors,
                    x='importance',
                    y='feature_name',
                    orientation='h',
                    color_discrete_sequence=['#e74c3c']
                )
                fig_risk.update_layout(
                    height=300,
                    showlegend=False,
                    xaxis_title="Impact on Risk",
                    yaxis_title="",
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig_risk, width='stretch')
            else:
                st.info("No significant risk-increasing factors identified.")
        
        with col_safe:
            st.markdown("### 🟢 Factors Decreasing Risk")
            safe_factors = feature_importance[feature_importance['importance'] < 0].nsmallest(7, 'importance')
            if len(safe_factors) > 0:
                safe_factors_plot = safe_factors.copy()
                safe_factors_plot['importance'] = safe_factors_plot['importance'].abs()
                fig_safe = px.bar(
                    safe_factors_plot,
                    x='importance',
                    y='feature_name',
                    orientation='h',
                    color_discrete_sequence=['#27ae60']
                )
                fig_safe.update_layout(
                    height=300,
                    showlegend=False,
                    xaxis_title="Positive Impact",
                    yaxis_title="",
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig_safe, width='stretch')
            else:
                st.info("No significant risk-decreasing factors identified.")

        # Feature comparison section
        st.markdown("---")
        st.markdown("## 📊 Client vs Population Comparison")
        
        # Key numeric features to compare
        compare_features = ['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'CREDIT_INCOME_RATIO', 
                          'EXT_SOURCE_MEAN', 'AGE_YEARS', 'EMPLOYMENT_YEARS']
        compare_features = [f for f in compare_features if f in X.columns]
        
        selected_feature = st.selectbox(
            "Select factor to compare",
            compare_features,
            format_func=format_feature_name
        )
        
        col_hist, col_stat = st.columns([3, 1])
        
        with col_hist:
            fig_dist = go.Figure()
            
            # Population distribution
            pop_data = X[selected_feature].dropna()
            fig_dist.add_trace(go.Histogram(
                x=pop_data,
                name='All Clients',
                opacity=0.7,
                nbinsx=50,
                marker_color='#3498db'
            ))
            
            # Client value
            client_val = client_data[selected_feature].values[0]
            fig_dist.add_vline(
                x=client_val, 
                line_width=3, 
                line_dash="dash", 
                line_color="#e74c3c",
                annotation_text=f"This Client: {client_val:.2f}"
            )
            
            fig_dist.update_layout(
                title=f"Distribution of {format_feature_name(selected_feature)}",
                xaxis_title=format_feature_name(selected_feature),
                yaxis_title="Number of Clients",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig_dist, width='stretch')
        
        with col_stat:
            st.markdown("### Statistics")
            pop_mean = pop_data.mean()
            pop_median = pop_data.median()
            percentile = (pop_data < client_val).mean() * 100
            
            st.metric("Client Value", f"{client_val:.2f}")
            st.metric("Population Mean", f"{pop_mean:.2f}")
            st.metric("Population Median", f"{pop_median:.2f}")
            st.metric("Percentile", f"{percentile:.0f}th")

        # Expandable detailed view
        with st.expander("📋 View Complete Client Profile"):
            if client_raw is not None:
                display_cols = ['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 
                              'CREDIT_INCOME_RATIO', 'AGE_YEARS', 'EMPLOYMENT_YEARS',
                              'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
                display_cols = [c for c in display_cols if c in client_raw.columns]
                
                profile_data = client_raw[display_cols].T
                profile_data.index = profile_data.index.map(format_feature_name)
                profile_data.columns = ['Value']
                st.dataframe(profile_data, width='stretch')

elif model_loaded:
    # Welcome state
    st.markdown("""
    <div style='text-align: center; padding: 60px 20px; background: white; border-radius: 15px; 
                box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin: 20px 0;'>
        <h1 style='font-size: 72px; margin: 0;'>🏦</h1>
        <h2 style='color: #1a365d; margin: 20px 0;'>Welcome to Credit Risk Assessment</h2>
        <p style='color: #666; font-size: 18px; max-width: 600px; margin: 0 auto;'>
            Enter a Client ID in the sidebar and click <strong>Analyze Risk Profile</strong> 
            to generate a comprehensive credit assessment powered by machine learning.
        </p>
        <br>
        <p style='color: #999;'>
            Model Performance: <strong>AUC 0.76</strong> | Features: <strong>290</strong> | 
            Training Samples: <strong>307,511</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown("---")
    st.markdown("### 🎯 Dashboard Capabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='stat-card'>
            <h3>🔍 Risk Assessment</h3>
            <p>AI-powered default probability prediction with confidence scoring</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='stat-card'>
            <h3>📊 Explainability</h3>
            <p>SHAP-based feature importance showing key risk drivers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='stat-card'>
            <h3>📈 Comparison</h3>
            <p>Compare client metrics against the entire population</p>
        </div>
        """, unsafe_allow_html=True)
