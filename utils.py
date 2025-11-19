import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# ----------------------------
# Dataset loading and bias metrics
# ----------------------------

def load_dataset(path):
    df = pd.read_csv(path)

    # Remove unnamed or blank columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Auto-detect target column
    possible_targets = ["grade", "label", "score", "marks", "outcome", "result", "target", "y"]
    target = None
    for col in df.columns:
        if col.lower() in possible_targets or df[col].dtype in [np.int64, np.float64]:
            target = col
            break

    if target is None:
        raise ValueError("No suitable numeric target column found. Add grade/score/label column.")

    # Auto-detect sensitive attributes (categorical with ≤10 unique values)
    sensitive_cols = []
    for col in df.columns:
        unique_vals = df[col].nunique()
        if unique_vals <= 10 and df[col].dtype == object:
            sensitive_cols.append(col)

    if not sensitive_cols:
        raise ValueError("No valid sensitive attribute found (e.g., gender, region, category).")

    return df, target, sensitive_cols


def basic_bias_metrics(df, target, sensitive):
    result = {}
    for col in sensitive:
        groups = df[col].unique()
        result[col] = {}
        for g in groups:
            subset = df[df[col] == g][target]
            result[col][str(g)] = {
                "count": len(subset),
                "mean_score": round(subset.mean(), 2),
                "std_dev": round(subset.std(), 2)
            }
    return result

# ----------------------------
# Preprocessing & feature pipeline
# ----------------------------

def basic_preprocess(df, target_col, sensitive_cols):
    df = df.copy()
    # Fill missing categorical values
    for col in sensitive_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
    # Fill missing numeric target
    if target_col not in df.columns:
        df[target_col] = 0
    df[target_col] = df[target_col].fillna(0)
    return df


def build_feature_pipeline(df, sensitive_cols):
    # Numeric features excluding sensitive columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in sensitive_cols]

    # Categorical features excluding sensitive columns
    categorical_cols = [c for c in df.columns if df[c].dtype == object and c not in sensitive_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(), categorical_cols)
        ],
        remainder="drop"
    )
    # Return the transformer and feature lists for reference
    return preprocessor, numeric_cols + categorical_cols, preprocessor

# ----------------------------
# Fairness / bias metrics
# ----------------------------

def positive_rate(y, sensitive_values, sensitive_col):
    rates = {}
    for group in np.unique(sensitive_values):
        idx = (sensitive_values == group)
        rates[group] = np.mean(y[idx])
    return rates


def demographic_parity_difference_from_rates(rates):
    if not rates:
        return None
    return max(rates.values()) - min(rates.values())


def disparate_impact_ratio_from_rates(rates):
    if not rates:
        return None
    min_rate = min(rates.values())
    max_rate = max(rates.values())
    return min_rate / max_rate if max_rate != 0 else 0


def tpr_by_group(y_true, y_pred, sensitive_values):
    tprs = {}
    for group in np.unique(sensitive_values):
        idx = (sensitive_values == group)
        true_pos = np.sum((y_true[idx] == 1) & (y_pred[idx] == 1))
        pos_total = np.sum(y_true[idx] == 1)
        tprs[group] = true_pos / pos_total if pos_total > 0 else 0
    return tprs

# ----------------------------
# Reweighing weights for bias mitigation
# ----------------------------

def compute_reweighing_weights(df, sensitive_col, label_col="label"):
    weights = pd.Series(1, index=df.index, dtype=float)
    groups = df[sensitive_col].unique()
    labels = df[label_col].unique()

    # Compute P(Y=y) and P(S=s)
    p_y = df[label_col].value_counts(normalize=True)
    p_s = df[sensitive_col].value_counts(normalize=True)

    # Compute P(Y=y, S=s) and assign weight w = P(Y=y)*P(S=s)/P(Y=y,S=s)
    for s in groups:
        for y in labels:
            idx = (df[sensitive_col] == s) & (df[label_col] == y)
            p_ys = idx.mean()  # P(Y=y,S=s)
            if p_ys > 0:
                weights[idx] = (p_y[y] * p_s[s]) / p_ys
    return weights
import matplotlib
matplotlib.use("Agg")  # Needed for non-GUI servers
import matplotlib.pyplot as plt
import base64
from io import BytesIO

import plotly.graph_objects as go

def plot_rates(rates, title="Rates by Group"):
    groups = list(rates.keys())
    values = list(rates.values())
    
    fig = go.Figure(go.Bar(x=groups, y=values, text=values, textposition='auto'))
    fig.update_layout(title=title, template="plotly_dark", height=450)
    
    return fig.to_html(full_html=False)

