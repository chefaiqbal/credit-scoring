import streamlit as st
import pandas as pd
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Set page config
st.set_page_config(
    page_title="Home Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        background-color: #f5f5f5;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        background-color: #2980b9;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #3498db;
    }
</style>
""", unsafe_allow_html=True)

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "results" / "model"

# Feature Name Mapping
FEATURE_MAPPING = {
    "EXT_SOURCE_1": "External Score 1",
    "EXT_SOURCE_2": "External Score 2",
    "EXT_SOURCE_3": "External Score 3",
    "AMT_CREDIT": "Credit Amount",
    "AMT_ANNUITY": "Loan Annuity",
    "AMT_INCOME_TOTAL": "Total Income",
    "DAYS_BIRTH": "Age (Days)",
    "DAYS_EMPLOYED": "Employment Duration (Days)",
    "AMT_GOODS_PRICE": "Goods Price",
    "NAME_CONTRACT_TYPE": "Contract Type",
    "CODE_GENDER": "Gender",
    "FLAG_OWN_CAR": "Owns Car",
    "FLAG_OWN_REALTY": "Owns Realty",
    "CNT_CHILDREN": "Child Count",
    "NAME_EDUCATION_TYPE": "Education Level",
    "NAME_FAMILY_STATUS": "Family Status",
    "NAME_HOUSING_TYPE": "Housing Type",
    "REGION_POPULATION_RELATIVE": "Region Population",
    "DAYS_REGISTRATION": "Registration Days",
    "DAYS_ID_PUBLISH": "ID Publish Days",
    "OWN_CAR_AGE": "Car Age",
    "CNT_FAM_MEMBERS": "Family Members",
    "REGION_RATING_CLIENT": "Region Rating",
    "REGION_RATING_CLIENT_W_CITY": "Region Rating (City)",
    "HOUR_APPR_PROCESS_START": "Application Hour",
    "REG_CITY_NOT_LIVE_CITY": "Reg City != Live City",
    "REG_CITY_NOT_WORK_CITY": "Reg City != Work City",
    "LIVE_CITY_NOT_WORK_CITY": "Live City != Work City",
    "ORGANIZATION_TYPE": "Organization Type",
    "OBS_30_CNT_SOCIAL_CIRCLE": "Social Circle Obs (30)",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Social Circle Def (30)",
    "OBS_60_CNT_SOCIAL_CIRCLE": "Social Circle Obs (60)",
    "DEF_60_CNT_SOCIAL_CIRCLE": "Social Circle Def (60)",
    "DAYS_LAST_PHONE_CHANGE": "Days Since Phone Change",
    "AMT_REQ_CREDIT_BUREAU_HOUR": "Credit Inquiries (Hour)",
    "AMT_REQ_CREDIT_BUREAU_DAY": "Credit Inquiries (Day)",
    "AMT_REQ_CREDIT_BUREAU_WEEK": "Credit Inquiries (Week)",
    "AMT_REQ_CREDIT_BUREAU_MON": "Credit Inquiries (Month)",
    "AMT_REQ_CREDIT_BUREAU_QRT": "Credit Inquiries (Quarter)",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Credit Inquiries (Year)",
}

def format_feature_name(name):
    """Convert technical feature name to user-friendly name."""
    if name in FEATURE_MAPPING:
        return FEATURE_MAPPING[name]
    # Generic formatting: replace underscores with spaces and title case
    return name.replace('_', ' ').title()

@st.cache_resource
def load_model():
    """Load the trained pipeline."""
    model_path = MODEL_DIR / "baseline_logreg.joblib"
    return joblib.load(model_path)

@st.cache_data
def load_data():
    """Load train features for context."""
    # We need SK_ID_CURR to look up clients
    # Using a subset or full data depending on memory
    df = pd.read_csv(DATA_DIR / "application_train_features.csv")
    df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(int)
    df.set_index('SK_ID_CURR', inplace=True)
    return df

def align_features(df, pipeline):
    """Align features with model expectations."""
    scaler = pipeline.named_steps['standardscaler']
    
    # Identify categorical columns
    X = df.drop(columns=["TARGET"], errors='ignore')
    cat_cols = X.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)

    # Align features
    if hasattr(pipeline, "feature_names_in_"):
        expected_features = pipeline.feature_names_in_
    elif hasattr(scaler, "feature_names_in_"):
        expected_features = scaler.feature_names_in_
    else:
        expected_features = X.columns # Fallback

    # Reindex to match model features
    X = X.reindex(columns=expected_features, fill_value=0)
    X = X.replace([float("inf"), float("-inf")], pd.NA).fillna(0)
    return X, expected_features

# Load resources
with st.spinner('Initializing Banking System...'):
    try:
        pipeline = load_model()
        scaler = pipeline.named_steps['standardscaler']
        classifier = pipeline.named_steps['logisticregression']
        
        raw_df = load_data()
        X, feature_names = align_features(raw_df, pipeline)
        
        # Initialize SHAP Explainer (using a background sample)
        background_sample = X.sample(100, random_state=42)
        background_scaled = scaler.transform(background_sample)
        background_scaled_df = pd.DataFrame(background_scaled, columns=X.columns)
        explainer = shap.LinearExplainer(classifier, background_scaled_df)
        
    except Exception as e:
        st.error(f"System Error: {e}")
        st.stop()

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/bank-building.png", width=80)
st.sidebar.title("Home Credit Risk")
st.sidebar.markdown("---")
st.sidebar.header("Client Lookup")
client_id = st.sidebar.number_input("Client ID", value=int(raw_df.index[0]), help="Enter the unique Client ID (SK_ID_CURR)")

analyze_btn = st.sidebar.button("Analyze Risk Profile")

st.sidebar.markdown("---")
st.sidebar.info("This dashboard provides credit risk assessment based on historical data and machine learning models.")

# Main Content
st.title("Credit Risk Assessment Dashboard")

if analyze_btn:
    if client_id not in X.index:
        st.error(f"Client ID {client_id} not found in the database.")
    else:
        # Get client data
        client_data = X.loc[[client_id]]
        
        # Predict
        client_scaled = scaler.transform(client_data)
        prob = classifier.predict_proba(client_scaled)[0][1]
        
        # Layout: Score Card
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("Default Probability")
            
            # Color coding
            score_color = "green"
            risk_label = "Low Risk"
            if prob > 0.5:
                score_color = "red"
                risk_label = "High Risk"
            elif prob > 0.2:
                score_color = "orange"
                risk_label = "Medium Risk"
                
            st.markdown(f"<h1 style='color:{score_color}; font-size: 48px;'>{prob:.2%}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:{score_color};'>{risk_label}</h3>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("### Assessment Summary")
            st.write(f"Client **{client_id}** has a **{prob:.2%}** probability of default.")
            st.write("This assessment is based on the client's application data, credit history, and external sources.")
            st.progress(float(prob))

        st.markdown("---")

        # SHAP Analysis
        st.header("Key Risk Drivers")
        st.write("The following factors contributed most significantly to this risk score.")
        
        client_scaled_df = pd.DataFrame(client_scaled, columns=X.columns)
        shap_values = explainer.shap_values(client_scaled_df)
        
        # Handle shape
        sv = shap_values[0] if isinstance(shap_values, list) else shap_values
        if len(sv.shape) > 1:
            sv = sv[0]
            
        # Prepare data for plot
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': sv
        })
        
        # Map feature names
        feature_importance['feature_name'] = feature_importance['feature'].apply(format_feature_name)
        
        # Sort and take top 15
        feature_importance = feature_importance.sort_values(by='importance', key=abs, ascending=True).tail(15)
        
        # Plot
        fig_shap = px.bar(
            feature_importance, 
            x='importance', 
            y='feature_name', 
            orientation='h',
            title="Top Factors Influencing Score",
            labels={'importance': 'Impact on Risk Score', 'feature_name': 'Factor'},
            color='importance',
            color_continuous_scale=['#2ecc71', '#e74c3c'] # Green to Red
        )
        fig_shap.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig_shap, use_container_width=True)
        
        # Feature Distribution
        st.markdown("---")
        st.header("Factor Comparison")
        st.write("Compare this client's data against the general population.")
        
        # Dropdown with formatted names
        # Create a mapping for the dropdown
        # We need to map back to technical names for plotting
        
        # Get top features from SHAP for the dropdown list (most relevant first)
        top_features_tech = feature_importance.sort_values(by='importance', key=abs, ascending=False)['feature'].tolist()
        # Add some standard ones if not in top
        standard_features = ['EXT_SOURCE_3', 'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH']
        for f in standard_features:
            if f not in top_features_tech and f in X.columns:
                top_features_tech.append(f)
                
        # Create options list
        options = {format_feature_name(f): f for f in top_features_tech}
        
        selected_feature_name = st.selectbox("Select Factor", list(options.keys()))
        selected_feature_tech = options[selected_feature_name]
        
        col_dist1, col_dist2 = st.columns([3, 1])
        
        with col_dist1:
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=X[selected_feature_tech], 
                name='Population', 
                opacity=0.7, 
                nbinsx=30,
                marker_color='#3498db'
            ))
            
            client_val = client_data[selected_feature_tech].values[0]
            fig_dist.add_vline(x=client_val, line_width=3, line_dash="dash", line_color="#e74c3c", annotation_text="Client")
            
            fig_dist.update_layout(
                title=f"Distribution of {selected_feature_name}", 
                xaxis_title=selected_feature_name, 
                yaxis_title="Count",
                bargap=0.1
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
        with col_dist2:
            st.markdown(f"### Client Value")
            
            # Format display value based on feature type
            display_val = client_val
            suffix = ""
            
            if "DAYS" in selected_feature_tech:
                display_val = abs(client_val)
                if "BIRTH" in selected_feature_tech:
                    display_val = display_val / 365
                    suffix = " years"
                elif "EMPLOYED" in selected_feature_tech:
                    if display_val > 300000: # The magic number for unemployed/pensioner
                        display_val = 0
                        suffix = " (Pensioner/Unemployed)"
                    else:
                        display_val = display_val / 365
                        suffix = " years"
                else:
                    suffix = " days ago"
            
            st.metric(label=selected_feature_name, value=f"{display_val:,.2f}{suffix}")
            
            # Format mean similarly for context
            pop_mean = X[selected_feature_tech].mean()
            if "DAYS" in selected_feature_tech:
                pop_mean = abs(pop_mean)
                if "BIRTH" in selected_feature_tech or ("EMPLOYED" in selected_feature_tech and pop_mean < 300000):
                    pop_mean = pop_mean / 365
            
            st.write(f"Population Mean: {pop_mean:,.2f}")
        
        # Raw Data Expander
        with st.expander("View Detailed Client Record"):
            # Transpose for better readability
            client_display = client_data.T.copy()
            
            # Clean up DAYS columns in the detailed view
            for col in client_display.index:
                if "DAYS" in col:
                    val = client_display.loc[col].values[0]
                    if "BIRTH" in col:
                        client_display.loc[col] = f"{abs(val)/365:.1f} Years"
                    elif "EMPLOYED" in col:
                        if val > 300000:
                            client_display.loc[col] = "Pensioner/Unemployed"
                        else:
                            client_display.loc[col] = f"{abs(val)/365:.1f} Years"
                    else:
                        client_display.loc[col] = f"{abs(val):.0f} Days ago"
            
            client_display.index = client_display.index.map(format_feature_name)
            client_display.columns = ["Value"]
            st.dataframe(client_display)

else:
    # Welcome / Empty State
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h2>Welcome to the Credit Risk Assessment System</h2>
        <p>Please enter a Client ID in the sidebar to begin analysis.</p>
    </div>
    """, unsafe_allow_html=True)
