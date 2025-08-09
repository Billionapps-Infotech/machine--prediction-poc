import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta

import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier

# ---------- Runtime TMP dir for CatBoost (fixes "Can't create train tmp dir: tmp")
TMPDIR = os.environ.get("TMPDIR", "/app/tmp")
os.makedirs(TMPDIR, exist_ok=True)

# ---------- Model selection (env override: USE_CATBOOST=1|0)
_USE_CAT = os.getenv("USE_CATBOOST", "1") == "1"
try:
    from catboost import CatBoostClassifier
except Exception:
    _USE_CAT = False

st.set_page_config(page_title="Machine Failure Prediction POC", layout="wide")
st.title("🔧 Machine Failure Prediction — POC")
st.caption("Dataset: July 2024 → July 2025 · Forecast next 12 months")

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

def left_panel():
    st.sidebar.header("⬇️ Data (single dataset)")
    df = load_csv("data/sample_machine.csv")
    st.sidebar.download_button("Download sample CSV", df.to_csv(index=False).encode("utf-8"),
                               "sample_machine.csv", "text/csv")
    uploaded = st.sidebar.file_uploader("Upload CSV (same columns)", type=["csv"])
    if uploaded is not None:
        try:
            up = pd.read_csv(uploaded)
            required = {"date","month","machine_id","age_months","ambient_temp_c","load_pct",
                        "run_hours","vibration_rms","pressure_bar","temperature_c",
                        "maintenance_overdue","defect_label"}
            if required.issubset(up.columns):
                df = up.copy()
            else:
                st.sidebar.error("Uploaded CSV missing required columns. Using bundled sample.")
        except Exception as e:
            st.sidebar.error(f"Failed to read CSV: {e}")
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌡️ Seasonality control")
    summer_boost = st.sidebar.slider("Summer ambient boost (°C)", 0.0, 10.0, 4.0, 0.5)
    st.sidebar.markdown("---")
    diversity = st.sidebar.checkbox(
        "Ensure variety in demo (diversity booster)",
        value=True,
        help="If too many rows pick the same label in a month, flip a few to the 2nd-best class to keep examples varied."
    )
    return df, summer_boost, diversity

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
    c = Counter(y_enc); total = sum(c.values())
    return {cls: total/(len(c)*cnt) for cls, cnt in c.items()}

def _rf_model():
    return RandomForestClassifier(
        n_estimators=500, random_state=42, n_jobs=-1,
        max_features="sqrt", class_weight="balanced"
    )

def train_model(df):
    """Train CatBoost (preferred) with safe fallback to RandomForest."""
    df = df.sort_values(["machine_id","date"]).reset_index(drop=True)
    X = build_features(df); y = df["defect_label"].values
    le = LabelEncoder(); y_enc = le.fit_transform(y)

    feats = [
        "age_months","ambient_temp_c","load_pct","run_hours","vibration_rms",
        "pressure_bar","temperature_c","maintenance_overdue",
        "year","month_num","month_idx","month_sin","month_cos","machine_id"
    ]

    from collections import Counter
    c = Counter(y_enc); strat = y_enc if min(c.values())>=2 and len(c)>1 else None
    if strat is None:
        st.warning("Stratified split disabled due to rare classes.", icon="⚠️")

    X_train, X_test, y_train, y_test = train_test_split(
        X[feats], y_enc, test_size=0.25, random_state=42, stratify=strat
    )

    used_cat = False
    if _USE_CAT:
        try:
            w = _class_weights(y_train); class_weights=[w[i] for i in sorted(w.keys())]
            model = CatBoostClassifier(
                iterations=600, depth=6, learning_rate=0.08,
                loss_function="MultiClass", l2_leaf_reg=3.0,
                random_state=42, class_weights=class_weights,
                allow_writing_files=False,      # <- no tmp artifacts
                train_dir=TMPDIR,               # <- safe tmp path
                verbose=False
            )
            model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
            y_pred_test = np.argmax(model.predict_proba(X_test), axis=1)
            used_cat = True
        except Exception as e:
            st.warning(f"CatBoost unavailable or failed ({e}). Falling back to RandomForest.", icon="⚠️")

    if not used_cat:
        model = _rf_model()
        model.fit(X_train, y_train)
        # RF has predict_proba
        y_pred_test = model.predict(X_test)

    report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_test, labels=np.unique(y_enc))
    acc = (y_pred_test==y_test).mean()
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average="macro", zero_division=0)
    metrics = {"accuracy": acc, "macro_precision": p, "macro_recall": r, "macro_f1": f1}

    return model, le, feats, pd.DataFrame(report).T.reset_index().rename(columns={"index":"metric"}), cm, metrics, np.unique(y_enc)

def generate_future(df, months_ahead=12, summer_boost=4.0):
    last_date = pd.to_datetime(df["date"]).max()
    start = (last_date + relativedelta(months=1)).replace(day=1)
    future_months = [start + relativedelta(months=i) for i in range(months_ahead)]
    machines = df["machine_id"].unique(); rows=[]
    for m in machines:
        hist = df[df["machine_id"]==m]
        age0 = hist["age_months"].max(); load_mu = hist["load_pct"].mean()
        vib_mu = hist["vibration_rms"].mean(); press_mu = hist["pressure_bar"].mean()
        temp_mu = hist["temperature_c"].mean(); run_mu = hist["run_hours"].mean()
        amb_mu = hist["ambient_temp_c"].mean(); maint_p = max(0.08, min(0.35, hist["maintenance_overdue"].mean()))
        for i, d in enumerate(future_months, start=1):
            month = d.month; age_months = age0 + i
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
                "ambient_temp_c": round(ambient_temp_c,2),
                "load_pct": round(load_pct,3),
                "run_hours": round(run_hours,1),
                "vibration_rms": round(vibration_rms,3),
                "pressure_bar": round(pressure_bar,2),
                "temperature_c": round(temperature_c,2),
                "maintenance_overdue": maintenance_overdue
            })
    return pd.DataFrame(rows)

def _basis_from_prediction(row, label):
    reasons = []
    temp = row.get("temperature_c", 0)
    vib = row.get("vibration_rms", 0)
    load = row.get("load_pct", 0)
    press = row.get("pressure_bar", 0)
    maint = row.get("maintenance_overdue", 0)
    runh = row.get("run_hours", 0)
    try:
        mnum = int(pd.to_datetime(row.get("date", None)).month)
    except Exception:
        mnum = None

    if label == "Over Heating":
        if temp >= 95: reasons.append("very high operating temperature")
        if load >= 0.8: reasons.append("sustained high load")
        if maint == 1: reasons.append("maintenance overdue")
    elif label == "Electrical Fault":
        if load >= 0.75: reasons.append("high electrical load")
        if temp >= 90: reasons.append("elevated temperature")
        if runh >= 220: reasons.append("heavy run hours")
        if maint == 1 and len(reasons) < 3: reasons.append("maintenance overdue")
    elif label == "Hydraulic Failure":
        if press >= 60: reasons.append("high hydraulic pressure")
        if (mnum in (12,1,2)) and len(reasons) < 3: reasons.append("cold season")
        if maint == 1 and len(reasons) < 3: reasons.append("maintenance overdue")
    elif label == "Mechanical Wear":
        if vib >= 1.0: reasons.append("elevated vibration")
        if runh >= 240 and len(reasons) < 3: reasons.append("heavy run hours")
        if row.get("age_months", 0) >= 24 and len(reasons) < 3: reasons.append("advanced component age")
    else:  # No Failure
        healthy = []
        if temp < 90: healthy.append("normal temperature")
        if vib < 1.0: healthy.append("stable vibration")
        if press < 60: healthy.append("normal pressure")
        if not healthy: healthy = ["historical stability"]
        return ", ".join(healthy[:3])

    if not reasons and label != "No Failure":
        scores = [
            ("elevated vibration", (vib-0.8)),
            ("sustained high load", (load-0.7)*2),
            ("high hydraulic pressure", (press-58)/5),
            ("heavy run hours", (runh-220)/50),
            ("maintenance overdue", 0.5 if maint==1 else -1),
            ("elevated temperature", (temp-90)/10),
        ]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        reasons = [name for name, sc in scores if sc > 0][:2]

    if not reasons:
        reasons = ["historical trends & seasonality"]
    return ", ".join(reasons[:3])

def _format_top3(p, classes):
    order = np.argsort(p)[::-1][:3]
    return ", ".join([f"{classes[i]} ({p[i]*100:.0f}%)" for i in order])

def _style_rows_with_confidence(df_to_style, defect_col="Defect", conf_col="Max Prob"):
    def to_style(row):
        label = str(row.get(defect_col, "")); conf = float(row.get(conf_col, 0))
        if label != "No Failure":
            alpha = 0.15 + 0.55*min(1.0,max(0.15,conf))
            return [f"background-color: rgba(255,82,82,{alpha}); font-weight: 600;" for _ in row]
        else:
            alpha = 0.10 + 0.4*min(1.0,max(0.1,conf))
            return [f"background-color: rgba(46,204,113,{alpha}); font-weight: 500;" for _ in row]
    try:
        return df_to_style.style.apply(to_style, axis=1)
    except Exception:
        return df_to_style

# ---------- UI
df, summer_boost, diversity = left_panel()

st.markdown("### 📅 Forecast Horizon")
months = ["All months"] + list(pd.date_range(
    pd.to_datetime(df['date']).max() + pd.offsets.MonthBegin(1),
    periods=12, freq='MS'
).strftime("%b-%Y"))
chosen_month = st.selectbox("Select month to highlight", months, index=1)

with st.spinner("Training model and generating predictions..."):
    model, le, feats, report_df, cm, metrics, label_ids = train_model(df)
    future = generate_future(df, 12, summer_boost)
    Xf = build_features(future)
    proba = model.predict_proba(Xf[feats])
    classes = le.inverse_transform(np.arange(proba.shape[1]))
    y_idx = np.argmax(proba, axis=1)

def diversify(pred_idx, proba, months, threshold=0.6):
    pred_idx = pred_idx.copy()
    uniq_months = sorted(np.unique(months), key=lambda s: pd.to_datetime("01-"+s, format="%d-%b-%Y"))
    for m in uniq_months:
        sel = np.where(months==m)[0]
        if len(sel)==0: continue
        labels, counts = np.unique(pred_idx[sel], return_counts=True)
        top = labels[np.argmax(counts)]; frac = counts.max()/len(sel)
        if frac >= threshold and len(sel)>=3:
            dom_idxs = [i for i in sel if pred_idx[i]==top]
            confid = proba[dom_idxs, top]
            order = np.argsort(confid)[:min(2,len(dom_idxs))]
            for k in order:
                i = dom_idxs[k]
                second = np.argsort(proba[i])[::-1][1]
                pred_idx[i] = second
    return pred_idx

y_idx_div = diversify(y_idx, proba, Xf["month"].values) if diversity else y_idx
y_pred = classes[y_idx_div]

pred_df = Xf[["date","month","machine_id","temperature_c","vibration_rms",
              "load_pct","pressure_bar","run_hours","maintenance_overdue"]].copy()
pred_df["predicted_defect"] = y_pred
pred_df["max_prob"] = proba[np.arange(len(proba)), y_idx_div]
pred_df["top3"] = [_format_top3(proba[i], classes) for i in range(len(pred_df))]
pred_df["basis"] = pred_df.apply(lambda r: _basis_from_prediction(r, r["predicted_defect"]), axis=1)

with st.expander("📊 Model validation — confusion matrix & metrics", expanded=False):
    st.dataframe(report_df, use_container_width=True)
    labels_text = le.inverse_transform(label_ids)
    fig = go.Figure(data=go.Heatmap(z=cm, x=labels_text, y=labels_text, colorscale="Blues",
                                    text=cm, texttemplate="%{text}", showscale=True))
    fig.update_layout(title="Confusion Matrix (rows=actual, cols=predicted)",
                      xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig, use_container_width=True)
    st.write({k: round(v,3) for k,v in metrics.items()})

# ---------- Views
if chosen_month != "All months":
    st.markdown("### 🔮 Predictions by machine — " + chosen_month)
    view = pred_df[pred_df["month"]==chosen_month].sort_values("machine_id").copy()
    view["Machine Name"] = view["machine_id"].apply(lambda x: f"Machine {x}")
    view = view[["Machine Name","predicted_defect","top3","basis","max_prob"]].rename(
        columns={"predicted_defect":"Defect","top3":"Top probabilities",
                 "basis":"What this prediction is based on","max_prob":"Max Prob"})
    st.dataframe(_style_rows_with_confidence(view), use_container_width=True)
else:
    st.markdown("### 🔮 Predictions (next 12 months) — by month")
    for m in sorted(pred_df["month"].unique(),
                    key=lambda s: pd.to_datetime("01-"+s, format="%d-%b-%Y")):
        st.markdown(f"**{m}**")
        subset = pred_df[pred_df["month"]==m].sort_values("machine_id").copy()
        subset["Machine Name"] = subset["machine_id"].apply(lambda x: f"Machine {x}")
        view = subset[["Machine Name","predicted_defect","top3","basis","max_prob"]].rename(
            columns={"predicted_defect":"Defect","top3":"Top probabilities",
                     "basis":"What this prediction is based on","max_prob":"Max Prob"})
        st.dataframe(_style_rows_with_confidence(view), use_container_width=True)

# ---------- Preview (bound/independent)
st.markdown("### 📄 Preview source data (Jul 2024 → Jul 2025)")
mode = st.radio("Preview scope", ["Follow selection above", "All months", "Choose a month"],
                horizontal=True, index=0)
order = st.radio("Sort by month", ["Ascending", "Descending"], horizontal=True, index=0)
preview = df.copy(); preview["date"] = pd.to_datetime(preview["date"])

if mode == "Follow selection above" and chosen_month != "All months":
    if chosen_month not in preview["month"].unique():
        try:
            target_m = pd.to_datetime("01-"+chosen_month, format="%d-%b-%Y").month
            candidates = preview[preview["date"].dt.month==target_m]["month"].unique().tolist()
            if candidates:
                mapped = max([pd.to_datetime("01-"+c, format="%d-%b-%Y") for c in candidates]).strftime("%b-%Y")
                st.caption(f"Showing historical month **{mapped}** (no source data for {chosen_month}).")
                preview = preview[preview["month"]==mapped]
        except Exception:
            pass
    else:
        preview = preview[preview["month"]==chosen_month]
elif mode == "Choose a month":
    months_src = sorted(preview["month"].unique().tolist(),
                        key=lambda s: pd.to_datetime("01-"+s, format="%d-%b-%Y"))
    pick = st.selectbox("Select month to preview", months_src, index=0)
    preview = preview[preview["month"]==pick]
# else All months -> no filter

preview = preview.sort_values("date", ascending=(order=="Ascending")).reset_index(drop=True)
st.dataframe(preview, use_container_width=True)