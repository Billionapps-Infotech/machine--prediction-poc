
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta

import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier

# Try CatBoost (better multiclass) and fall back to RF
_USE_CAT = True
try:
    from catboost import CatBoostClassifier
except Exception:
    _USE_CAT = False

# Optional OpenAI explanations
_USE_OPENAI = False
try:
    from openai import OpenAI
    if os.getenv("OPENAI_API_KEY"):
        _USE_OPENAI = True
        _openai_client = OpenAI()
except Exception:
    _USE_OPENAI = False

st.set_page_config(page_title="Machine Failure Prediction POC", layout="wide")
st.title("🔧 Machine Failure Prediction — POC")
st.caption("Dataset: July 2024 → July 2025 · Forecast next 12 months")

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

def left_panel():
    st.sidebar.header("⬇️ Data (single dataset)")
    df = load_csv("data/sample_machine.csv")
    st.sidebar.download_button(
        label="Download sample CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="sample_machine.csv",
        mime="text/csv"
    )
    uploaded = st.sidebar.file_uploader("Upload CSV (same columns)", type=["csv"])
    if uploaded is not None:
        try:
            up = pd.read_csv(uploaded)
            required = {"date","month","machine_id","age_months","ambient_temp_c","load_pct","run_hours","vibration_rms","pressure_bar","temperature_c","maintenance_overdue","defect_label"}
            if required.issubset(up.columns):
                df = up.copy()
            else:
                st.sidebar.error("Uploaded CSV missing required columns. Using bundled sample.")
        except Exception as e:
            st.sidebar.error(f"Failed to read CSV: {e}")
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌡️ Seasonality control")
    summer_boost = st.sidebar.slider("Summer ambient boost (°C)", min_value=0.0, max_value=10.0, value=4.0, step=0.5,
                                     help="Adds extra heat to Jul/Aug during forecasting to simulate site ambient temperature.")
    return df, summer_boost

def build_features(df):
    X = df.copy()
    X["date"] = pd.to_datetime(X["date"])
    X["year"] = X["date"].dt.year
    X["month_num"] = X["date"].dt.month
    X["month_idx"] = (X["date"].dt.year - X["date"].dt.year.min())*12 + X["date"].dt.month
    X["month_sin"] = np.sin(2*np.pi*X["month_num"]/12)
    X["month_cos"] = np.cos(2*np.pi*X["month_num"]/12)
    return X

def _class_weights(y_enc):
    from collections import Counter
    c = Counter(y_enc)
    total = sum(c.values())
    return {cls: total/(len(c)*cnt) for cls, cnt in c.items()}

def train_model(df):
    df = df.sort_values(["machine_id","date"]).reset_index(drop=True)
    X = build_features(df)
    y = df["defect_label"].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    features = [
        "age_months","ambient_temp_c","load_pct","run_hours","vibration_rms",
        "pressure_bar","temperature_c","maintenance_overdue",
        "year","month_num","month_idx","month_sin","month_cos","machine_id"
    ]
    from collections import Counter
    counts = Counter(y_enc)
    strat = y_enc if min(counts.values()) >= 2 and len(counts) > 1 else None
    if strat is None:
        st.warning("Stratified split disabled due to rare classes.", icon="⚠️")
    X_train, X_test, y_train, y_test = train_test_split(
        X[features], y_enc, test_size=0.25, shuffle=True, random_state=42, stratify=strat
    )
    if _USE_CAT:
        w = _class_weights(y_train)
        class_weights = [w[i] for i in sorted(w.keys())]
        model = CatBoostClassifier(
            iterations=600, depth=6, learning_rate=0.08, loss_function="MultiClass",
            l2_leaf_reg=3.0, random_state=42, class_weights=class_weights, verbose=False
        )
        model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
        y_pred_test = np.argmax(model.predict_proba(X_test), axis=1)
    else:
        model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1, max_features="sqrt", class_weight="balanced")
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
    # Metrics
    report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index":"metric"})
    cm = confusion_matrix(y_test, y_pred_test, labels=np.unique(y_enc))
    acc = accuracy_score(y_test, y_pred_test)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average="macro", zero_division=0)
    metrics = {"accuracy": acc, "macro_precision": p, "macro_recall": r, "macro_f1": f1}
    return model, le, features, report_df, cm, metrics, np.unique(y_enc)

def generate_future(df, months_ahead=12, summer_boost=4.0):
    last_date = pd.to_datetime(df["date"]).max()
    start = (last_date + relativedelta(months=1)).replace(day=1)
    future_months = [start + relativedelta(months=i) for i in range(months_ahead)]
    machines = df["machine_id"].unique()
    rows = []
    for m in machines:
        hist = df[df["machine_id"]==m]
        age0 = hist["age_months"].max()
        load_mu = hist["load_pct"].mean()
        vib_mu = hist["vibration_rms"].mean()
        press_mu = hist["pressure_bar"].mean()
        temp_mu = hist["temperature_c"].mean()
        run_mu = hist["run_hours"].mean()
        amb_mu = hist["ambient_temp_c"].mean()
        maint_p = max(0.08, min(0.35, hist["maintenance_overdue"].mean()))
        for i, d in enumerate(future_months, start=1):
            month = d.month
            age_months = age0 + i
            heat = summer_boost if month in (7,8) else 0.0
            ambient_temp_c = amb_mu + 4*np.sin((month-1)/12*2*np.pi) + heat
            load_pct = np.clip(load_mu + 0.1*np.sin(i/6*np.pi) + (0.08 if month in (7,8) else 0.0), 0, 1)
            run_hours = np.clip(run_mu + 25*np.sin(i/6*np.pi), 60, 360)
            vibration_rms = np.clip(vib_mu + 0.02*i + 0.05*np.sin(i/6*np.pi), 0, None)
            pressure_bar = np.clip(press_mu + 3*np.sin(i/6*np.pi), 20, 110)
            temperature_c = np.clip(temp_mu + 0.45*load_pct*100 + 0.7*np.sin(i/6*np.pi) + heat, 40, 140)
            maintenance_overdue = int(np.random.random() < maint_p)
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "month": d.strftime("%b-%Y"),
                "machine_id": m,
                "age_months": age_months,
                "ambient_temp_c": round(ambient_temp_c, 2),
                "load_pct": round(load_pct, 3),
                "run_hours": round(run_hours, 1),
                "vibration_rms": round(vibration_rms, 3),
                "pressure_bar": round(pressure_bar, 2),
                "temperature_c": round(temperature_c, 2),
                "maintenance_overdue": maintenance_overdue,
            })
    return pd.DataFrame(rows)

def _basis_from_features(row):
    basis = []
    try:
        if row.get("temperature_c", 0) >= 90:
            basis.append("high operating temperature")
        if row.get("vibration_rms", 0) >= 1.0:
            basis.append("elevated vibration")
        if row.get("load_pct", 0) >= 0.8:
            basis.append("sustained high load")
        if row.get("pressure_bar", 0) >= 60:
            basis.append("high hydraulic pressure")
        if row.get("maintenance_overdue", 0) == 1:
            basis.append("maintenance overdue")
        if row.get("run_hours", 0) >= 220:
            basis.append("heavy run hours")
    except Exception:
        pass
    if not basis:
        basis = ["historical trends & seasonality"]
    return ", ".join(basis)

def _style_rows_with_confidence(df_to_style, defect_col="Defect", conf_col="Max Prob"):
    def to_style(row):
        label = str(row.get(defect_col, ""))
        conf = float(row.get(conf_col, 0))  # 0..1
        if label != "No Failure":
            # red gradient based on confidence
            intensity = min(1.0, max(0.15, conf))
            color = f"rgba(255, 82, 82, {0.15 + 0.55*intensity})"
            weight = "700" if conf >= 0.8 else "600"
            return [f"background-color: {color}; font-weight: {weight};" for _ in row]
        else:
            # green gradient for no failure (confidence of being safe)
            intensity = min(1.0, max(0.1, conf))
            color = f"rgba(46, 204, 113, {0.10 + 0.4*intensity})"
            weight = "600" if conf >= 0.8 else "500"
            return [f"background-color: {color}; font-weight: {weight};" for _ in row]
    try:
        return df_to_style.style.apply(to_style, axis=1)
    except Exception:
        return df_to_style

df, summer_boost = left_panel()

st.markdown("### 📅 Forecast Horizon")
months = ["All months"] + list(pd.date_range(pd.to_datetime(df['date']).max() + pd.offsets.MonthBegin(1), periods=12, freq='MS').strftime("%b-%Y"))
chosen_month = st.selectbox("Select month to highlight", months, index=1)

with st.spinner("Training model and generating predictions..."):
    model, le, features, report_df, cm, metrics, label_ids = train_model(df)
    future = generate_future(df, months_ahead=12, summer_boost=summer_boost)
    X_future = build_features(future)
    if _USE_CAT:
        proba = model.predict_proba(X_future[features])
    else:
        proba = model.predict_proba(X_future[features])
    classes = le.inverse_transform(np.arange(proba.shape[1]))
    y_pred_idx = np.argmax(proba, axis=1)
    y_pred = classes[y_pred_idx]

def _format_top3(probs_row, classes):
    order = np.argsort(probs_row)[::-1][:3]
    items = [f"{classes[i]} ({probs_row[i]*100:.0f}%)" for i in order]
    return ", ".join(items)

pred_df = X_future[["date","month","machine_id","temperature_c","vibration_rms","load_pct","pressure_bar","run_hours","maintenance_overdue"]].copy()
pred_df["predicted_defect"] = y_pred
pred_df["max_prob"] = proba.max(axis=1)
pred_df["top3"] = [_format_top3(proba[i], classes) for i in range(len(pred_df))]
pred_df["basis"] = pred_df.apply(lambda r: _basis_from_features(r), axis=1)

# --- Confusion Matrix card ---
with st.expander("📊 Model validation — confusion matrix & metrics", expanded=False):
    st.dataframe(report_df, use_container_width=True)
    labels_text = le.inverse_transform(label_ids)
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels_text, y=labels_text,
        colorscale="Blues", text=cm, texttemplate="%{text}", showscale=True
    ))
    fig.update_layout(title="Confusion Matrix (rows=actual, cols=predicted)", xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig, use_container_width=True)
    st.write({k: round(v, 3) for k, v in metrics.items()})

# --- Views ---
if chosen_month != "All months":
    st.markdown("### 🔮 Predictions by machine — " + chosen_month)
    view = pred_df[pred_df["month"] == chosen_month].sort_values("machine_id").copy()
    view["Machine Name"] = view["machine_id"].apply(lambda x: f"Machine {x}")
    view = view[["Machine Name","predicted_defect","top3","basis","max_prob"]].rename(
        columns={"predicted_defect":"Defect","top3":"Top probabilities","basis":"What this prediction is based on","max_prob":"Max Prob"}
    ).reset_index(drop=True)
    st.dataframe(_style_rows_with_confidence(view), use_container_width=True)
else:
    st.markdown("### 🔮 Predictions (next 12 months) — by month")
    months_order = sorted(pred_df["month"].unique().tolist(), key=lambda s: pd.to_datetime("01-" + s, format="%d-%b-%Y"))
    for m in months_order:
        st.markdown(f"**{m}**")
        subset = pred_df[pred_df["month"] == m].sort_values("machine_id").copy()
        subset["Machine Name"] = subset["machine_id"].apply(lambda x: f"Machine {x}")
        view = subset[["Machine Name","predicted_defect","top3","basis","max_prob"]].rename(
            columns={"predicted_defect":"Defect","top3":"Top probabilities","basis":"What this prediction is based on","max_prob":"Max Prob"}
        ).reset_index(drop=True)
        st.dataframe(_style_rows_with_confidence(view), use_container_width=True)

# --- Preview at bottom ---
st.markdown("### 📄 Preview source data (Jul 2024 → Jul 2025)")
order = st.radio("Sort by month", ["Ascending", "Descending"], horizontal=True, index=0)
preview = df.copy()
preview["date"] = pd.to_datetime(preview["date"])
preview = preview.sort_values("date", ascending=(order=="Ascending"))
st.dataframe(preview.head(50), use_container_width=True)
