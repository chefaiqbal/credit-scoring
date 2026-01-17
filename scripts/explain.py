"""
Explainability Script for Credit Scoring Model
==============================================

This script provides comprehensive model interpretability including:
1. SHAP force plots for individual predictions
2. Plotly visualizations comparing clients to population
3. Detailed written analysis of correct/wrong predictions
4. Feature importance analysis

"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments

import matplotlib.pyplot as plt
import pandas as pd
import shap
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "results" / "model"
OUTPUT_DIR = BASE_DIR / "results" / "clients_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature name mapping for readability
FEATURE_MAPPING = {
    "EXT_SOURCE_1": "External Score 1",
    "EXT_SOURCE_2": "External Score 2",
    "EXT_SOURCE_3": "External Score 3",
    "AMT_CREDIT": "Credit Amount",
    "AMT_ANNUITY": "Loan Annuity",
    "AMT_INCOME_TOTAL": "Total Income",
    "DAYS_BIRTH": "Age (Days)",
    "DAYS_EMPLOYED": "Employment Duration",
    "AMT_GOODS_PRICE": "Goods Price",
    "bureau_row_count": "Bureau Records Count",
    "prev_row_count": "Previous Applications",
}


def format_feature_name(name):
    """Convert technical feature name to readable format."""
    if name in FEATURE_MAPPING:
        return FEATURE_MAPPING[name]
    return name.replace('_', ' ').title()[:40]


def load_data():
    """Load train and test features with SK_ID_CURR preserved."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    # Load train features
    train_df = pd.read_csv(DATA_DIR / "application_train_features.csv")
    
    # Preserve SK_ID_CURR for client identification
    sk_id_train = train_df["SK_ID_CURR"].values if "SK_ID_CURR" in train_df.columns else None
    
    if "TARGET" in train_df.columns:
        y_train = train_df["TARGET"]
        X_train = train_df.drop(columns=["TARGET"])
    else:
        X_train = train_df
        y_train = None
    
    # Load test features
    test_df = pd.read_csv(DATA_DIR / "application_test_features.csv")
    sk_id_test = test_df["SK_ID_CURR"].values if "SK_ID_CURR" in test_df.columns else None
    
    if "TARGET" in test_df.columns:
        X_test = test_df.drop(columns=["TARGET"])
    else:
        X_test = test_df
    
    # Store raw data before encoding for visualization
    X_train_raw = X_train.copy()
    X_test_raw = X_test.copy()
    
    # Handle categorical columns (One-Hot Encoding)
    cat_cols = X_train.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        print(f"Encoding {len(cat_cols)} categorical columns...")
        X_train = pd.get_dummies(X_train, columns=cat_cols, dummy_na=True)
        X_test = pd.get_dummies(X_test, columns=cat_cols, dummy_na=True)
    
    # Align columns with model expectations
    model_path = MODEL_DIR / "baseline_logreg.joblib"
    try:
        pipeline = joblib.load(model_path)
        if hasattr(pipeline, "feature_names_in_"):
            expected_features = pipeline.feature_names_in_
        elif hasattr(pipeline.steps[0][1], "feature_names_in_"):
            expected_features = pipeline.steps[0][1].feature_names_in_
        else:
            expected_features = None
        
        if expected_features is not None:
            print(f"Aligning to {len(expected_features)} model features...")
            X_train = X_train.reindex(columns=expected_features, fill_value=0)
            X_test = X_test.reindex(columns=expected_features, fill_value=0)
    except Exception as e:
        print(f"Warning: Feature alignment issue: {e}")
    
    # Clean data
    X_train = X_train.replace([float("inf"), float("-inf")], np.nan).fillna(0)
    X_test = X_test.replace([float("inf"), float("-inf")], np.nan).fillna(0)
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    return X_train, y_train, X_test, sk_id_train, sk_id_test, X_train_raw, X_test_raw


def load_model():
    """Load the trained pipeline."""
    model_path = MODEL_DIR / "baseline_logreg.joblib"
    print(f"Loading model from {model_path}...")
    return joblib.load(model_path)


def find_interesting_clients(classifier, X_train_scaled, y_train, probs):
    """
    Find clients that demonstrate correct and wrong predictions.
    
    Strategy:
    - CORRECT: High confidence prediction that matches true label
    - WRONG: Confident wrong prediction (most interesting for analysis)
    """
    preds = (probs > 0.5).astype(int)
    y_array = y_train.values if hasattr(y_train, 'values') else y_train
    
    correct_mask = preds == y_array
    wrong_mask = preds != y_array
    
    # For CORRECT: Find a defaulter (TARGET=1) correctly predicted with high confidence
    correct_default_mask = correct_mask & (y_array == 1) & (probs > 0.6)
    if correct_default_mask.sum() > 0:
        # Pick the one with highest probability
        correct_idx = np.where(correct_default_mask)[0]
        correct_client_idx = correct_idx[np.argmax(probs[correct_idx])]
    else:
        # Fallback: any correct prediction
        correct_client_idx = np.where(correct_mask)[0][0]
    
    # For WRONG: Find a case where model was confident but wrong
    # Priority: Predicted low risk but actually defaulted (False Negative - dangerous!)
    wrong_fn_mask = wrong_mask & (y_array == 1) & (probs < 0.4)  # Predicted safe, was risky
    if wrong_fn_mask.sum() > 0:
        wrong_idx = np.where(wrong_fn_mask)[0]
        # Pick the one with lowest probability (most confident wrong)
        wrong_client_idx = wrong_idx[np.argmin(probs[wrong_idx])]
    else:
        # Fallback: any wrong prediction
        wrong_indices = np.where(wrong_mask)[0]
        if len(wrong_indices) > 0:
            wrong_client_idx = wrong_indices[0]
        else:
            wrong_client_idx = 1
    
    return correct_client_idx, wrong_client_idx


def create_client_comparison_plot(client_data, population_data, client_id, features_to_compare, output_path):
    """
    Create Plotly visualization comparing client to population.
    Required by project spec: "comparison between this client and other clients"
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[format_feature_name(f) for f in features_to_compare[:4]]
    )
    
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for idx, feature in enumerate(features_to_compare[:4]):
        if feature not in population_data.columns:
            continue
        
        row, col = positions[idx]
        
        # Population histogram
        pop_values = population_data[feature].dropna()
        fig.add_trace(
            go.Histogram(x=pop_values, name="Population", opacity=0.7, 
                        marker_color='#3498db', showlegend=(idx == 0)),
            row=row, col=col
        )
        
        # Client value as vertical line
        client_val = client_data[feature].values[0]
        fig.add_vline(
            x=client_val, line_width=3, line_dash="dash", 
            line_color="#e74c3c", row=row, col=col
        )
    
    fig.update_layout(
        title=f"Client {client_id}: Feature Comparison vs Population",
        height=600,
        showlegend=True
    )
    
    fig.write_html(output_path)
    print(f"  Saved comparison plot: {output_path}")


def create_shap_bar_plot_plotly(shap_values, feature_names, client_id, output_path):
    """Create Plotly bar chart of SHAP values (top factors)."""
    
    # Handle shap_values shape
    sv = shap_values[0] if isinstance(shap_values, list) else shap_values
    if len(sv.shape) > 1:
        sv = sv[0]
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': sv
    })
    
    # Sort by absolute value and take top 15
    importance_df['abs_shap'] = importance_df['shap_value'].abs()
    importance_df = importance_df.nlargest(15, 'abs_shap').sort_values('shap_value')
    importance_df['feature_name'] = importance_df['feature'].apply(format_feature_name)
    
    # Color by positive/negative impact
    importance_df['color'] = importance_df['shap_value'].apply(
        lambda x: '#e74c3c' if x > 0 else '#2ecc71'  # Red = increases risk, Green = decreases risk
    )
    
    fig = px.bar(
        importance_df,
        x='shap_value',
        y='feature_name',
        orientation='h',
        title=f"Client {client_id}: Top Factors Influencing Risk Score",
        labels={'shap_value': 'Impact on Risk (SHAP Value)', 'feature_name': 'Factor'},
        color='shap_value',
        color_continuous_scale=['#2ecc71', '#f5f5f5', '#e74c3c']
    )
    
    fig.update_layout(height=500, showlegend=False)
    fig.write_html(output_path)
    print(f"  Saved SHAP bar plot: {output_path}")
    
    return importance_df


def analyze_wrong_prediction(client_data, client_raw, shap_values, feature_names, 
                             true_label, predicted_prob, population_data):
    """
    Detailed analysis of why the model got a prediction wrong.
    This is the KEY requirement: "Try to understand why the model got wrong on this client"
    """
    sv = shap_values[0] if isinstance(shap_values, list) else shap_values
    if len(sv.shape) > 1:
        sv = sv[0]
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': sv
    }).sort_values('shap_value', key=abs, ascending=False)
    
    analysis = []
    analysis.append("=" * 70)
    analysis.append("ANALYSIS: WHY THE MODEL GOT THIS PREDICTION WRONG")
    analysis.append("=" * 70)
    analysis.append("")
    analysis.append(f"True Outcome: {'DEFAULTED (1)' if true_label == 1 else 'REPAID (0)'}")
    analysis.append(f"Model Predicted Probability of Default: {predicted_prob:.2%}")
    analysis.append(f"Model Decision: {'HIGH RISK' if predicted_prob > 0.5 else 'LOW RISK'}")
    analysis.append("")
    
    if true_label == 1 and predicted_prob < 0.5:
        analysis.append("ERROR TYPE: FALSE NEGATIVE (Missed a defaulter)")
        analysis.append("This is a DANGEROUS error - the model approved a risky client.")
        analysis.append("")
        analysis.append("FACTORS THAT MISLED THE MODEL (pushed toward 'safe'):")
        analysis.append("-" * 50)
        
        # Features that pushed prediction toward 0 (safe) but shouldn't have
        misleading = importance_df[importance_df['shap_value'] < 0].head(5)
        for _, row in misleading.iterrows():
            feat = row['feature']
            shap_val = row['shap_value']
            if feat in client_raw.columns:
                client_val = client_raw[feat].values[0]
                pop_mean = population_data[feat].mean() if feat in population_data.columns else np.nan
                analysis.append(f"  • {format_feature_name(feat)}")
                if not np.isnan(pop_mean):
                    analysis.append(f"    Client Value: {client_val:.4f}, Population Mean: {pop_mean:.4f}")
                else:
                    analysis.append(f"    Client Value: {client_val:.4f}")
                analysis.append(f"    SHAP Impact: {shap_val:.4f} (pushed toward SAFE)")
        
    elif true_label == 0 and predicted_prob > 0.5:
        analysis.append("ERROR TYPE: FALSE POSITIVE (Flagged a good client)")
        analysis.append("The model was overly cautious and rejected a good client.")
        analysis.append("")
        analysis.append("FACTORS THAT MISLED THE MODEL (pushed toward 'risky'):")
        analysis.append("-" * 50)
        
        misleading = importance_df[importance_df['shap_value'] > 0].head(5)
        for _, row in misleading.iterrows():
            feat = row['feature']
            shap_val = row['shap_value']
            if feat in client_raw.columns:
                client_val = client_raw[feat].values[0]
                pop_mean = population_data[feat].mean() if feat in population_data.columns else np.nan
                analysis.append(f"  • {format_feature_name(feat)}")
                if not np.isnan(pop_mean):
                    analysis.append(f"    Client Value: {client_val:.4f}, Population Mean: {pop_mean:.4f}")
                else:
                    analysis.append(f"    Client Value: {client_val:.4f}")
                analysis.append(f"    SHAP Impact: {shap_val:.4f} (pushed toward RISKY)")
    
    analysis.append("")
    analysis.append("CONCLUSION:")
    analysis.append("-" * 50)
    analysis.append("The model relies heavily on external scores (EXT_SOURCE) and employment")
    analysis.append("data. This client likely had atypical patterns that don't fit the linear")
    analysis.append("assumptions of logistic regression. A tree-based model might capture")
    analysis.append("such non-linear interactions better.")
    analysis.append("")
    
    return "\n".join(analysis)


def generate_client_report(client_name, client_id, client_data, client_raw, 
                           shap_values, feature_names, population_data,
                           true_label=None, predicted_prob=None, is_wrong=False):
    """Generate comprehensive report for a single client."""
    
    report = []
    report.append("=" * 70)
    report.append(f"CLIENT REPORT: {client_name}")
    report.append(f"SK_ID_CURR: {client_id}")
    report.append("=" * 70)
    report.append("")
    
    if predicted_prob is not None:
        report.append(f"Predicted Default Probability: {predicted_prob:.2%}")
        report.append(f"Risk Classification: {'HIGH RISK' if predicted_prob > 0.5 else 'LOW RISK'}")
    
    if true_label is not None:
        report.append(f"Actual Outcome: {'DEFAULTED' if true_label == 1 else 'REPAID'}")
        if is_wrong:
            report.append("*** MODEL PREDICTION: INCORRECT ***")
        else:
            report.append("Model Prediction: CORRECT")
    
    report.append("")
    report.append("KEY CLIENT CHARACTERISTICS:")
    report.append("-" * 40)
    
    # Show key features
    key_features = ['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 
                   'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
                   'DAYS_BIRTH', 'DAYS_EMPLOYED']
    
    for feat in key_features:
        if feat in client_raw.columns:
            val = client_raw[feat].values[0]
            if 'DAYS_BIRTH' in feat:
                report.append(f"  Age: {abs(val)/365:.1f} years")
            elif 'DAYS_EMPLOYED' in feat:
                if val > 300000:
                    report.append(f"  Employment: Pensioner/Unemployed")
                else:
                    report.append(f"  Employment: {abs(val)/365:.1f} years")
            else:
                report.append(f"  {format_feature_name(feat)}: {val:,.2f}")
    
    report.append("")
    
    return "\n".join(report)


def run_explanation():
    """Main execution function - comprehensive model explanation."""
    
    print("\n" + "=" * 70)
    print("CREDIT SCORING MODEL EXPLAINABILITY ANALYSIS")
    print("Senior ML Scientist Review")
    print("=" * 70 + "\n")
    
    # Load data
    X_train, y_train, X_test, sk_id_train, sk_id_test, X_train_raw, X_test_raw = load_data()
    model = load_model()
    
    pipeline = model
    scaler = pipeline.named_steps['standardscaler']
    classifier = pipeline.named_steps['logisticregression']
    
    # Scale data
    print("\nScaling data...")
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Get predictions
    print("Computing predictions...")
    subset_size = min(5000, len(X_train_scaled_df))
    train_probs = classifier.predict_proba(X_train_scaled_df.iloc[:subset_size])[:, 1]
    
    # Find interesting clients
    print("Selecting clients for analysis...")
    correct_idx, wrong_idx = find_interesting_clients(
        classifier, 
        X_train_scaled_df.iloc[:subset_size],
        y_train.iloc[:subset_size],
        train_probs
    )
    test_idx = 0  # First test client
    
    # Get client IDs
    correct_client_id = sk_id_train[correct_idx] if sk_id_train is not None else correct_idx
    wrong_client_id = sk_id_train[wrong_idx] if sk_id_train is not None else wrong_idx
    test_client_id = sk_id_test[test_idx] if sk_id_test is not None else test_idx
    
    print(f"\nSelected Clients:")
    print(f"  CORRECT: Index {correct_idx}, SK_ID_CURR={correct_client_id}")
    print(f"    True: {y_train.iloc[correct_idx]}, Prob: {train_probs[correct_idx]:.2%}")
    print(f"  WRONG:   Index {wrong_idx}, SK_ID_CURR={wrong_client_id}")
    print(f"    True: {y_train.iloc[wrong_idx]}, Prob: {train_probs[wrong_idx]:.2%}")
    print(f"  TEST:    Index {test_idx}, SK_ID_CURR={test_client_id}")
    
    # Initialize SHAP explainer
    print("\nInitializing SHAP LinearExplainer...")
    explainer = shap.LinearExplainer(classifier, X_train_scaled_df)
    
    # Key features for comparison plots
    comparison_features = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'AMT_CREDIT', 'AMT_INCOME_TOTAL',
                          'DAYS_BIRTH', 'AMT_ANNUITY', 'DAYS_EMPLOYED', 'AMT_GOODS_PRICE']
    comparison_features = [f for f in comparison_features if f in X_train_raw.columns]
    
    # Master report
    master_report = []
    master_report.append("=" * 70)
    master_report.append("CREDIT SCORING MODEL - CLIENT ANALYSIS REPORT")
    master_report.append("=" * 70)
    master_report.append("")
    
    # Process each client
    clients = [
        ("client_correct_train", correct_idx, X_train_scaled_df, X_train_raw, 
         y_train.iloc[correct_idx], train_probs[correct_idx], correct_client_id, False),
        ("client_wrong_train", wrong_idx, X_train_scaled_df, X_train_raw,
         y_train.iloc[wrong_idx], train_probs[wrong_idx], wrong_client_id, True),
        ("client_test", test_idx, X_test_scaled_df, X_test_raw,
         None, None, test_client_id, False),
    ]
    
    for name, idx, scaled_df, raw_df, true_label, prob, client_id, is_wrong in clients:
        print(f"\n{'=' * 60}")
        print(f"Processing: {name} (SK_ID_CURR={client_id})")
        print("=" * 60)
        
        client_scaled = scaled_df.iloc[[idx]]
        client_raw = raw_df.iloc[[idx]]
        
        # Get probability for test client
        if prob is None:
            prob = classifier.predict_proba(client_scaled)[0][1]
        
        # SHAP values
        print("  Computing SHAP values...")
        shap_values = explainer.shap_values(client_scaled)
        
        # 1. SHAP Force Plot (matplotlib)
        print("  Generating SHAP force plot...")
        shap.force_plot(
            explainer.expected_value,
            shap_values,
            client_scaled,
            matplotlib=True,
            show=False
        )
        plt.savefig(OUTPUT_DIR / f"{name}_force_plot.png", bbox_inches='tight', dpi=150)
        plt.close()
        
        # 2. SHAP Bar Plot (Plotly) - Required by project
        print("  Generating Plotly SHAP bar chart...")
        importance_df = create_shap_bar_plot_plotly(
            shap_values, 
            X_train.columns.tolist(),
            client_id,
            OUTPUT_DIR / f"{name}_shap_importance.html"
        )
        
        # 3. Client vs Population Comparison (Plotly) - Required by project
        print("  Generating population comparison plot...")
        create_client_comparison_plot(
            client_raw,
            raw_df,
            client_id,
            comparison_features,
            OUTPUT_DIR / f"{name}_comparison.html"
        )
        
        # 4. Generate text report
        report = generate_client_report(
            name, client_id, client_scaled, client_raw,
            shap_values, X_train.columns.tolist(), raw_df,
            true_label, prob, is_wrong
        )
        master_report.append(report)
        
        # 5. Special analysis for wrong prediction
        if is_wrong:
            print("  Analyzing wrong prediction...")
            wrong_analysis = analyze_wrong_prediction(
                client_scaled, client_raw, shap_values,
                X_train.columns.tolist(), true_label, prob, raw_df
            )
            master_report.append(wrong_analysis)
            
            # Save wrong analysis separately
            with open(OUTPUT_DIR / "wrong_prediction_analysis.txt", 'w') as f:
                f.write(wrong_analysis)
            print(f"  Saved analysis: wrong_prediction_analysis.txt")
    
    # Save master report
    master_report_text = "\n".join(master_report)
    with open(OUTPUT_DIR / "client_analysis_report.txt", 'w') as f:
        f.write(master_report_text)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    run_explanation()
