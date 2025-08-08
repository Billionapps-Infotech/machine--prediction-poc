
import os
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from dateutil.relativedelta import relativedelta

import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Machine Failure Prediction POC", layout="wide")
st.title("🔧 Machine Failure Prediction — POC")
st.caption("Dataset: July 2024 → July 2025 · Forecast next 12 months")


def _style_defect_rows(df_to_style, defect_col="Defect"):
    import pandas as _pd
    import numpy as _np
    def row_style(row):
        is_defect = str(row.get(defect_col, "")) != "No Failure"
        if is_defect:
            return ["background-color: #ffe9e9; font-weight: 600;" for _ in row]
        else:
            return ["" for _ in row]
    try:
        return df_to_style.style.apply(row_style, axis=1)
    except Exception:
        return df_to_style

def _preview_right(df_original):
    st.markdown("### 📄 Preview source data (Jul 2024 → Jul 2025)")
    order = st.radio("Sort by month", ["Ascending", "Descending"], horizontal=True, index=0)
    view = df_original.copy()
    view["date"] = pd.to_datetime(view["date"])
    view = view.sort_values("date", ascending=(order=="Ascending"))
    st.dataframe(view.head(50), use_container_width=True)


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
    with st.sidebar.expander("Preview", expanded=False):
        st.dataframe(df.head(15), use_container_width=True)
    return df

def build_features(df):
    X = df.copy()
    X["date"] = pd.to_datetime(X["date"])
    X["year"] = X["date"].dt.year
    X["month_num"] = X["date"].dt.month
    X["month_idx"] = (X["date"].dt.year - X["date"].dt.year.min())*12 + X["date"].dt.month
    X["month_sin"] = np.sin(2*np.pi*X["month_num"]/12)
    X["month_cos"] = np.cos(2*np.pi*X["month_num"]/12)
    return X

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
    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, max_features="sqrt")
    model.fit(X_train, y_train)
    report = classification_report(y_test, model.predict(X_test), output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index":"metric"})
    return model, le, features, report_df

def generate_future(df, months_ahead=12):
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
            ambient_temp_c = amb_mu + 3*np.sin((month-1)/12*2*np.pi)
            load_pct = np.clip(load_mu + 0.08*np.sin(i/6*np.pi), 0, 1)
            run_hours = np.clip(run_mu + 20*np.sin(i/6*np.pi), 60, 340)
            vibration_rms = np.clip(vib_mu + 0.02*i + 0.07*np.sin(i/6*np.pi), 0, None)
            pressure_bar = np.clip(press_mu + 3*np.sin(i/6*np.pi), 20, 110)
            temperature_c = np.clip(temp_mu + 0.4*load_pct*100 + 0.8*np.sin(i/6*np.pi), 40, 135)
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

def _short_explanation(row, label):
    drivers = _basis_from_features(row)
    return f"{label} expected due to {drivers}."


def explain(X_future):
    explanations = []
    for i in range(len(X_future)):
        row = X_future.iloc[i]
        key_feats = ["temperature_c","vibration_rms","load_pct","pressure_bar","run_hours"]
        parts = [f"{k}={round(row[k],2)}" for k in key_feats if k in row]
        explanations.append("Drivers: " + ", ".join(parts[:3]))
    return explanations

df = left_panel()

st.markdown("### 📅 Forecast Horizon")
months = ["All months"] + list(pd.date_range(pd.to_datetime(df['date']).max() + pd.offsets.MonthBegin(1), periods=12, freq='MS').strftime("%b-%Y"))
chosen_month = st.selectbox("Select month to highlight", months, index=1)

with st.spinner("Training model and generating predictions..."):
    model, le, features, report_df = train_model(df)
    future = generate_future(df, months_ahead=12)
    X_future = build_features(future)
    proba = model.predict_proba(X_future[features])
    classes = le.inverse_transform(np.arange(proba.shape[1]))
    y_pred_idx = np.argmax(proba, axis=1)
    y_pred = classes[y_pred_idx]
    explanations = explain(X_future)

pred_df = X_future[["date","month","machine_id","temperature_c","vibration_rms","load_pct","pressure_bar","run_hours","maintenance_overdue"]].copy()
pred_df["predicted_defect"] = y_pred
pred_df["basis"] = pred_df.apply(lambda r: _basis_from_features(r), axis=1)
pred_df["explanation"] = pred_df.apply(lambda r: _short_explanation(r, r["predicted_defect"]), axis=1)

if chosen_month != "All months":
    st.markdown("### 🔮 Predictions by machine — " + chosen_month)
    view = pred_df[pred_df["month"] == chosen_month].sort_values("machine_id").copy()
    view["Machine Name"] = view["machine_id"].apply(lambda x: f"Machine {x}")
    view = view[["Machine Name","predicted_defect","explanation","basis"]].rename(
        columns={"predicted_defect":"Defect","explanation":"Short Explanation","basis":"What this prediction is based on"}
    )
    st.dataframe(_style_defect_rows(view.reset_index(drop=True)), use_container_width=True)
else:
    st.markdown("### 🔮 Predictions (next 12 months) — by month")
    months_order = sorted(pred_df["month"].unique().tolist(), key=lambda s: pd.to_datetime("01-" + s, format="%d-%b-%Y"))
    for m in months_order:
        st.markdown(f"**{m}**")
        subset = pred_df[pred_df["month"] == m].sort_values("machine_id").copy()
        subset["Machine Name"] = subset["machine_id"].apply(lambda x: f"Machine {x}")
        view = subset[["Machine Name","predicted_defect","explanation","basis"]].rename(
            columns={"predicted_defect":"Defect","explanation":"Short Explanation","basis":"What this prediction is based on"}
        ).reset_index(drop=True)
        st.dataframe(_style_defect_rows(view), use_container_width=True)

count_by_month = pred_df.groupby(["month","predicted_defect"]).size().reset_index(name="count")
fig = px.bar(count_by_month, x="month", y="count", color="predicted_defect", barmode="stack", title="Predicted defects per month")
st.plotly_chart(fig, use_container_width=True)


# --- Right-side final section: preview original data ---
_preview_right(df)
