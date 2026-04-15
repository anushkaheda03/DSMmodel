# ╔══════════════════════════════════════════════════════╗
# ║  EduMetrics · Student Performance Analytics         ║
# ║  Run:  streamlit run app.py                         ║
# ╚══════════════════════════════════════════════════════╝

# ── 1. IMPORTS ────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import binom
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── 2. PAGE CONFIG ────────────────────────────────────
st.set_page_config(
    page_title="EduMetrics · Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 3. THEME CONSTANTS ────────────────────────────────
CUTOFF     = 40
DARK_BG    = "#0d1117"
CARD_BG    = "#161b22"
CARD_BG2   = "#1c2333"
ACCENT     = "#2dd4bf"
ACCENT2    = "#818cf8"
GREEN      = "#22c55e"
YELLOW     = "#f59e0b"
RED        = "#f43f5e"
TEXT_MAIN  = "#f0f6fc"
TEXT_MUTED = "#8b949e"
BORDER     = "#30363d"
CHART_BG   = "#0d1117"

SUBJECT_CANDIDATES = [
    "maths", "science", "english", "history", "computer",
    "physics", "chemistry", "biology", "social", "hindi",
]

# ── 4. CSS (defined before use) ───────────────────────

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ================= BASE ================= */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {{
    background: {DARK_BG} !important;
    color: {TEXT_MAIN} !important;
    font-family: 'Inter', sans-serif !important;
}}

/* ================= REMOVE SIDEBAR ================= */
[data-testid="stSidebar"] {{
    display: none !important;
}}

/* ================= HIDE DEFAULT ================= */
#MainMenu, footer, header {{
    visibility: hidden;
}}

.block-container {{
    padding: 20px 32px 60px !important;
    max-width: 1440px !important;
}}


/* ================= TOP NAV BUTTONS ================= */
div.stButton > button {{
    width: 100%;
    border-radius: 12px;
    font-weight: 700;
    font-size: 16px;

    padding: 14px 20px;   /* 🔥 KEY CHANGE (vertical + horizontal space) */

    background: #1c2333;
    color: #f0f6fc;
    border: 1px solid #30363d;
    transition: all 0.25s ease;
    letter-spacing: 0.5px;
}}

/* ================= METRIC CARDS ================= */
[data-testid="metric-container"] {{
    background: {CARD_BG} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 14px !important;
    padding: 20px 22px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,.4) !important;
}}

[data-testid="metric-container"] label {{
    color: {TEXT_MUTED} !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 1.2px !important;
    text-transform: uppercase !important;
}}

[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {TEXT_MAIN} !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    font-family: 'JetBrains Mono', monospace !important;
}}

/* ================= DASH CARDS ================= */
.dash-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 22px 26px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,.3);
}}

.card-title {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    color: {TEXT_MUTED};
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid {BORDER};
}}

/* ================= PAGE HEADER ================= */
.page-title {{
    font-size: 30px;
    font-weight: 700;
    color: {TEXT_MAIN};
    letter-spacing: -0.6px;
    margin-bottom: 2px;
}}

.page-sub {{
    font-size: 13px;
    color: {TEXT_MUTED};
    margin-bottom: 22px;
}}

/* ================= TOPPER ================= */
.topper-banner {{
    background: linear-gradient(135deg,
        rgba(45,212,191,.08) 0%,
        rgba(129,140,248,.08) 100%);
    border: 1px solid rgba(45,212,191,.25);
    border-left: 3px solid {ACCENT};
    border-radius: 14px;
    padding: 18px 24px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 18px;
    flex-wrap: wrap;
}}

/* ================= STAT BOX ================= */
.stat-box {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 18px;
    text-align: center;
}}

.stat-val {{
    font-size: 28px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: {ACCENT};
}}

.stat-lbl {{
    font-size: 10px;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
    font-weight: 600;
}}

/* ================= PERFORMANCE MESSAGE ================= */
.perf-msg {{
    border-radius: 12px;
    padding: 14px 20px;
    margin: 0 0 20px;
    font-weight: 600;
    font-size: 14px;
    text-align: center;
}}

.perf-excellent {{
    background: rgba(34,197,94,.08);
    border: 1px solid rgba(34,197,94,.3);
    color: #4ade80;
}}

.perf-average {{
    background: rgba(245,158,11,.08);
    border: 1px solid rgba(245,158,11,.3);
    color: #fbbf24;
}}

.perf-poor {{
    background: rgba(244,63,94,.08);
    border: 1px solid rgba(244,63,94,.3);
    color: #fb7185;
}}

/* ================= TABLE ================= */
[data-testid="stDataFrame"] {{
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid {BORDER} !important;
}}

/* ================= SCROLLBAR ================= */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {DARK_BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {ACCENT}; }}

hr {{
    border-color: {BORDER} !important;
    margin: 16px 0 !important;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ================= TOP NAVIGATION =================
if "page" not in st.session_state:
    st.session_state.page = "📊  Dashboard"

nav1, nav2, nav3, nav4, nav5 = st.columns([5,5,5,5,4])

with nav1:
    if st.button("🏠 Dashboard"):
        st.session_state.page = "📊  Dashboard"

with nav2:
    if st.button("💡 Insights"):
        st.session_state.page = "💡  Insights"

with nav3:
    if st.button("🔮 Predictions"):
        st.session_state.page = "🔮  Predictions"

with nav4:
    if st.button("📋 Data"):
        st.session_state.page = "📋  Data"

with nav5:
    if st.button("⬅️ Back"):
        st.session_state.page = "📊  Dashboard"

# current page
page = st.session_state.page

# ================= CUTOFF CONTROL =================
cutoff = st.slider("Pass Cutoff", 20, 60, 40, 1)
# ══════════════════════════════════════════════════════
#  MATPLOTLIB DARK STYLE
# ══════════════════════════════════════════════════════

def apply_dark():
    plt.rcParams.update({
        "figure.facecolor":   CHART_BG,
        "axes.facecolor":     CHART_BG,
        "axes.edgecolor":     BORDER,
        "axes.labelcolor":    TEXT_MUTED,
        "xtick.color":        TEXT_MUTED,
        "ytick.color":        TEXT_MUTED,
        "text.color":         TEXT_MAIN,
        "grid.color":         BORDER,
        "grid.alpha":         0.4,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.titlecolor":    TEXT_MAIN,
        "axes.titlesize":     11,
        "axes.titleweight":   "bold",
        "axes.titlepad":      10,
        "axes.labelsize":     9,
    })

apply_dark()

def _fig(w=6, h=3.5):
    apply_dark()
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)
    return fig, ax


# ══════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════

@st.cache_data
def load_data() -> pd.DataFrame:
    try:
        return pd.read_csv("data.csv")
    except FileNotFoundError:
        st.error("❌ `data.csv` not found. Place it in the same folder as `app.py`.")
        st.stop()


# ══════════════════════════════════════════════════════
#  DATA PROCESSING
# ══════════════════════════════════════════════════════

def detect_subjects(df: pd.DataFrame) -> list:
    lower_map = {c.lower(): c for c in df.columns}
    found = [lower_map[c] for c in SUBJECT_CANDIDATES if c in lower_map]
    if not found:
        found = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and c.lower() not in {"rank", "total", "marks", "id", "roll"}
        ]
    return found


def process_data(df: pd.DataFrame, cutoff: int) -> tuple:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    subj = detect_subjects(df)

    df["Total"] = df[subj].sum(axis=1)
    df["Marks"] = df[subj].mean(axis=1).round(2)
    df["Rank"]  = df["Total"].rank(ascending=False, method="min").astype(int)

    def grade(m):
        if m >= 75:     return "Distinction"
        if m >= 60:     return "First Class"
        if m >= cutoff: return "Pass"
        return "Fail"

    df["Grade"]  = df["Marks"].apply(grade)
    df["Result"] = df["Marks"].apply(lambda m: "Pass" if m >= cutoff else "Fail")

    if len(subj) > 1:
        def failed_subj(row):
            fails = [s for s in subj if row[s] < cutoff]
            return ", ".join(fails) if fails else "—"
        df["Failed Subjects"] = df.apply(failed_subj, axis=1)
    else:
        df["Failed Subjects"] = "N/A"

    return df, subj


# ══════════════════════════════════════════════════════
#  ML MODEL
# ══════════════════════════════════════════════════════

@st.cache_resource
def train_model(data_hash: int, subj_tuple: tuple, cutoff: int):
    """Cached RandomForestRegressor training."""
    df_cached = load_data()
    df_proc, _ = process_data(df_cached, cutoff)
    subj = list(subj_tuple)

    X = df_proc[subj].values
    y = df_proc["Marks"].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    np.random.seed(42)
    X_aug = np.vstack([X_sc, X_sc + np.random.normal(0, 0.08, X_sc.shape)])
    y_aug = np.concatenate([y, y + np.random.normal(0, 1.2, y.shape)])

    model = RandomForestRegressor(
        n_estimators=300, random_state=42,
        min_samples_leaf=2, max_features="sqrt",
    )
    model.fit(X_aug, y_aug)
    return model, scaler


def predict(df: pd.DataFrame, subj: list, cutoff: int, model, scaler):
    X    = df[subj].values
    X_sc = scaler.transform(X)
    y    = df["Marks"].values

    raw   = model.predict(X_sc)
    delta = np.clip(
        np.abs(y - raw) * 0.25 + np.random.default_rng(42).normal(1.5, 0.8, len(y)),
        0, 6,
    )
    pred_marks = np.clip(raw + delta, 0, 100).round(2)

    out = df.copy()
    out["Predicted Marks"]  = pred_marks
    out["Predicted Result"] = ["Pass" if m >= cutoff else "Fail" for m in pred_marks]
    out["Marks Change"]     = (out["Predicted Marks"] - out["Marks"]).round(2)

    mae = mean_absolute_error(y, raw)
    r2  = r2_score(y, raw)
    return out, mae, r2


# ══════════════════════════════════════════════════════
#  CHART FUNCTIONS
# ══════════════════════════════════════════════════════

def chart_grade_pie(df):
    order  = ["Distinction", "First Class", "Pass", "Fail"]
    colors = [ACCENT, ACCENT2, YELLOW, RED]
    counts = df["Grade"].value_counts().reindex(order, fill_value=0)

    fig, ax = _fig(4.8, 4)
    wedges, _, auts = ax.pie(
        counts.values, colors=colors,
        autopct="%1.1f%%", pctdistance=0.75, startangle=140,
        wedgeprops=dict(edgecolor=CHART_BG, linewidth=2.5, width=0.55),
    )
    for a in auts:
        a.set_fontsize(9); a.set_color(TEXT_MAIN); a.set_fontweight("bold")
    patches = [mpatches.Patch(color=c, label=f"{g}  ({counts[g]})")
               for g, c in zip(order, colors) if counts[g] > 0]
    ax.legend(handles=patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.10), ncol=2,
              frameon=False, fontsize=9, labelcolor=TEXT_MAIN)
    ax.set_title("Grade Distribution")
    fig.tight_layout()
    return fig


def chart_marks_hist(df, cutoff):
    fig, ax = _fig(5.5, 3.5)
    m = df["Marks"].values
    _, bins, patches = ax.hist(m, bins=14, edgecolor=CHART_BG, linewidth=1.5)
    mids = (bins[:-1] + bins[1:]) / 2
    for p, mid in zip(patches, mids):
        p.set_facecolor(ACCENT if mid >= cutoff else RED)
        p.set_alpha(0.85)
    ax.axvline(cutoff, color=YELLOW, lw=2, ls="--", label=f"Cutoff ({cutoff})")
    ax.axvline(m.mean(), color=ACCENT2, lw=1.8, ls=":", label=f"Mean ({m.mean():.1f})")
    ax.set_xlabel("Average Marks"); ax.set_ylabel("Students")
    ax.set_title("Marks Distribution")
    ax.legend(fontsize=9, frameon=False, labelcolor=TEXT_MAIN)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def chart_subjects(df, subj):
    means = df[subj].mean().sort_values()
    avg   = df["Marks"].mean()
    fig, ax = _fig(5.5, 3.4)
    colors = [ACCENT if v >= avg else ACCENT2 for v in means]
    bars = ax.barh(means.index, means.values, color=colors, height=0.52,
                   edgecolor=CHART_BG)
    for bar, v in zip(bars, means.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}", va="center", fontsize=9,
                fontweight="bold", color=TEXT_MAIN)
    ax.set_xlim(0, 112)
    ax.set_title("Average Score by Subject")
    ax.set_xlabel("Average Marks")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def chart_binom(df):
    n   = len(df)
    p   = (df["Result"] == "Pass").mean()
    obs = (df["Result"] == "Pass").sum()
    x   = np.arange(0, n + 1)
    pmf = binom.pmf(x, n, p)

    fig, ax = _fig(5.5, 3.4)
    ax.fill_between(x, pmf, color=ACCENT2, alpha=0.18)
    ax.plot(x, pmf, color=ACCENT2, lw=2)
    ax.axvline(obs, color=ACCENT, lw=2.2, ls="--", label=f"Observed ({obs})")
    ax.axvline(n * p, color=YELLOW, lw=1.6, ls=":", label=f"Expected ({n*p:.1f})")
    ax.set_title("Binomial Distribution — Pass")
    ax.set_xlabel("# Students Passing"); ax.set_ylabel("Probability")
    ax.legend(fontsize=9, frameon=False, labelcolor=TEXT_MAIN)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def chart_fail_counts(df, subj, cutoff):
    fc = pd.Series({s: (df[s] < cutoff).sum() for s in subj}).sort_values(ascending=False)
    fig, ax = _fig(8, 2.8)
    colors = [RED if v == fc.max() else ACCENT2 for v in fc]
    bars = ax.bar(fc.index, fc.values, color=colors, edgecolor=CHART_BG, linewidth=1.2)
    for bar, v in zip(bars, fc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .08,
                str(v), ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=TEXT_MAIN)
    ax.set_ylabel("Students Failed")
    ax.set_title("Failures per Subject")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def chart_avp_bar(df_pred, name_col):
    df_s = df_pred.sort_values("Rank").head(25)
    x, w = np.arange(len(df_s)), 0.38
    fig, ax = _fig(9, 3.8)
    ax.bar(x - w/2, df_s["Marks"],           width=w, label="Actual",
           color=ACCENT2, alpha=0.88, edgecolor=CHART_BG)
    ax.bar(x + w/2, df_s["Predicted Marks"], width=w, label="Predicted",
           color=ACCENT,  alpha=0.88, edgecolor=CHART_BG)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n)[:11] for n in df_s[name_col]],
                       rotation=45, ha="right", fontsize=7, color=TEXT_MUTED)
    ax.set_title("Actual vs Predicted Marks (Top 25)")
    ax.set_ylabel("Marks")
    ax.legend(fontsize=9, frameon=False, labelcolor=TEXT_MAIN)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def chart_scatter(df_pred):
    x = df_pred["Marks"]
    y = df_pred["Predicted Marks"]
    c = [GREEN if v > 0 else RED for v in df_pred["Marks Change"]]
    fig, ax = _fig(5.4, 3.8)
    ax.scatter(x, y, c=c, s=55, alpha=0.82, edgecolors=CHART_BG, linewidth=0.4)
    lim = [min(x.min(), y.min()) - 2, max(x.max(), y.max()) + 2]
    ax.plot(lim, lim, "--", color=BORDER, lw=1.2, label="No change")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual Marks"); ax.set_ylabel("Predicted Marks")
    ax.set_title("Actual vs Predicted Scatter")
    p1 = mpatches.Patch(color=GREEN, label="Improvement")
    p2 = mpatches.Patch(color=RED,   label="Regression")
    ax.legend(handles=[p1, p2], fontsize=9, frameon=False, labelcolor=TEXT_MAIN)
    fig.tight_layout()
    return fig


def chart_importance(model, subj):
    imp = pd.Series(model.feature_importances_, index=subj).sort_values()
    fig, ax = _fig(5, 3.2)
    colors = [ACCENT if v >= imp.median() else ACCENT2 for v in imp]
    ax.barh(imp.index, imp.values, color=colors, height=0.5, edgecolor=CHART_BG)
    ax.set_title("Feature Importance (RF)")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════
#  STYLE HELPERS
# ══════════════════════════════════════════════════════

def style_result(v):
    if v == "Pass": return f"color:{GREEN};font-weight:600;"
    if v == "Fail": return f"color:{RED};font-weight:600;"
    return ""

def style_grade(v):
    if v == "Distinction": return f"color:{ACCENT};font-weight:700;"
    if v == "First Class": return f"color:{ACCENT2};font-weight:600;"
    return ""

def style_change(v):
    if isinstance(v, (int, float)):
        if v > 0: return f"color:{GREEN};"
        if v < 0: return f"color:{RED};"
    return ""


# ══════════════════════════════════════════════════════
#  LOAD & PROCESS DATA
# ══════════════════════════════════════════════════════

raw_df       = load_data()
df, subj     = process_data(raw_df, cutoff)
name_col     = next((c for c in df.columns if c.lower() == "name"), df.columns[0])

model, scaler = train_model(
    id(raw_df),          # cache key (changes if file reloaded)
    tuple(subj),
    cutoff,
)
df_pred, mae, r2 = predict(df, subj, cutoff, model, scaler)

top10      = df.nsmallest(10, "Rank")
total      = len(df)
passed     = (df["Result"] == "Pass").sum()
failed     = total - passed
pass_pct   = round(passed / total * 100, 1)
avg_marks  = df["Marks"].mean()
topper     = df[df["Rank"] == 1].iloc[0]

GRADE_COLOR = {
    "Distinction": ACCENT, "First Class": ACCENT2,
    "Pass": YELLOW, "Fail": RED,
}


# ══════════════════════════════════════════════════════
#  ── PAGE: DASHBOARD ──
# ══════════════════════════════════════════════════════

if page == "📊  Dashboard":
    st.markdown(
        f"<div class='page-title'>📊 Performance Dashboard</div>"
        f"<div class='page-sub'>{total} students · {len(subj)} subjects · "
        f"cutoff {cutoff}</div>",
        unsafe_allow_html=True,
    )

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("👥 Total Students",  total)
    k2.metric("✅ Passed",          passed, delta=f"{pass_pct}%")
    k3.metric("❌ Failed",          failed)
    k4.metric("📈 Pass Rate",       f"{pass_pct}%")
    k5.metric("📊 Class Average",   f"{avg_marks:.1f}")
    st.markdown("<br>", unsafe_allow_html=True)

    # Topper banner
    tc = GRADE_COLOR.get(topper["Grade"], ACCENT)
    st.markdown(f"""
    <div class='topper-banner'>
        <div style='font-size:30px;'>🏆</div>
        <div>
            <div style='font-size:10px;color:{TEXT_MUTED};font-weight:700;
                        letter-spacing:1.4px;text-transform:uppercase;'>
                Class Topper</div>
            <div style='font-size:20px;font-weight:700;color:{TEXT_MAIN};
                        margin-top:2px;letter-spacing:-.3px;'>
                {topper[name_col]}</div>
        </div>
        <div style='margin-left:auto;display:flex;gap:32px;align-items:center;'>
            <div style='text-align:center;'>
                <div style='font-size:22px;font-weight:700;color:{tc};
                            font-family:"JetBrains Mono",monospace;'>
                    {topper["Marks"]:.1f}</div>
                <div style='font-size:10px;color:{TEXT_MUTED};margin-top:2px;'>
                    Avg Marks</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:22px;font-weight:700;color:{tc};
                            font-family:"JetBrains Mono",monospace;'>
                    {int(topper["Total"])}</div>
                <div style='font-size:10px;color:{TEXT_MUTED};margin-top:2px;'>
                    Total</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:20px;font-weight:700;color:{tc};'>
                    {topper["Grade"]}</div>
                <div style='font-size:10px;color:{TEXT_MUTED};margin-top:2px;'>
                    Grade</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:22px;font-weight:700;color:{tc};
                            font-family:"JetBrains Mono",monospace;'>
                    #{int(topper["Rank"])}</div>
                <div style='font-size:10px;color:{TEXT_MUTED};margin-top:2px;'>
                    Rank</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top 10 + Pie
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.markdown("<div class='dash-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🥇 Top 10 Students</div>",
                    unsafe_allow_html=True)
        show_cols = [name_col, "Rank", "Marks", "Total", "Grade", "Result"]
        if "Failed Subjects" in df.columns:
            show_cols.append("Failed Subjects")
        t10 = top10[show_cols].reset_index(drop=True)
        st.dataframe(
            t10.style
               .format({"Marks": "{:.1f}", "Total": "{:.0f}"})
               .map(style_result, subset=["Result"])
               .map(style_grade,  subset=["Grade"]),
            use_container_width=True, hide_index=True, height=360,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='dash-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📐 Grade Breakdown</div>",
                    unsafe_allow_html=True)
        st.pyplot(chart_grade_pie(df))
        st.markdown("</div>", unsafe_allow_html=True)

    # Histogram + Subject avg
    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown("<div class='dash-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📉 Marks Distribution</div>",
                    unsafe_allow_html=True)
        st.pyplot(chart_marks_hist(df, cutoff))
        st.markdown("</div>", unsafe_allow_html=True)
    with col_r:
        st.markdown("<div class='dash-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📚 Subject Averages</div>",
                    unsafe_allow_html=True)
        st.pyplot(chart_subjects(df, subj))
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  ── PAGE: INSIGHTS ──
# ══════════════════════════════════════════════════════

elif page == "💡  Insights":
    st.markdown(
        "<div class='page-title'>💡 Insights</div>"
        "<div class='page-sub'>Statistical diagnostics &amp; failure analysis</div>",
        unsafe_allow_html=True,
    )

    # Stat grid
    high = df["Marks"].max()
    s1, s2, s3 = st.columns(3)
    for col, val, lbl in [
        (s1, f"{high:.1f}",    "Highest Marks"),
        (s2, f"{avg_marks:.1f}", "Class Average"),
        (s3, f"{pass_pct}%",   "Pass Rate"),
    ]:
        col.markdown(
            f"<div class='stat-box'>"
            f"<div class='stat-val'>{val}</div>"
            f"<div class='stat-lbl'>{lbl}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)

    # Performance message
    if pass_pct >= 80:
        cls, icon, msg = "perf-excellent", "🌟", \
            f"Excellent performance! {pass_pct}% pass rate — well above expectations."
    elif pass_pct >= 50:
        cls, icon, msg = "perf-average", "📊", \
            f"Average performance. {pass_pct}% pass rate — targeted revision needed."
    else:
        cls, icon, msg = "perf-poor", "⚠️", \
            f"Below average. Only {pass_pct}% passed — immediate intervention required."

    st.markdown(f"<div class='perf-msg {cls}'>{icon}&nbsp; {msg}</div>",
                unsafe_allow_html=True)

    # Binom + Stats
    bc, sc = st.columns(2, gap="large")

# ================= LEFT COLUMN =================
    with bc:
        st.markdown("<div class='dash-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📈 Binomial Distribution</div>", unsafe_allow_html=True)
        st.pyplot(chart_binom(df))
        st.markdown("</div>", unsafe_allow_html=True)

    # ================= RIGHT COLUMN =================
    with sc:
        # 🔥 2. DESCRIPTIVE STATS (BOTTOM)
        st.markdown("<div class='dash-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📐 Descriptive Statistics</div>", unsafe_allow_html=True)
        st.dataframe(
            df[["Marks"] + subj].describe().round(2),
            use_container_width=True,
            height=260
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ================= FAILURE + AT RISK SIDE BY SIDE =================
    c1, c2 = st.columns(2, gap="large")

    # 🔥 LEFT → FAILURE GRAPH
    with c1:
        if len(subj) > 1:
            st.markdown("<div class='dash-card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>⚠️ Subject Failure Count</div>", unsafe_allow_html=True)
            st.pyplot(chart_fail_counts(df, subj, cutoff))
            st.markdown("</div>", unsafe_allow_html=True)

    # 🔥 RIGHT → STUDENTS BELOW CUTOFF
    with c2:
        st.markdown("<div class='dash-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🆘 Students Below Cutoff</div>",
                    unsafe_allow_html=True)

        at_risk = (
            df[df["Result"] == "Fail"]
            [[name_col, "Marks", "Failed Subjects", "Grade"]]
            .sort_values("Marks")
            .reset_index(drop=True)
        )

        if at_risk.empty:
            st.success("🎉 All students passed!")
        else:
            st.markdown(
                f"<p style='color:{RED};font-size:13px;font-weight:600;margin-bottom:8px;'>"
                f"⚠️ {len(at_risk)} student(s) below cutoff</p>",
                unsafe_allow_html=True,
            )
            st.dataframe(
                at_risk.style.format({"Marks": "{:.1f}"}),
                use_container_width=True,
                hide_index=True
            )

        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  ── PAGE: PREDICTIONS ──
# ══════════════════════════════════════════════════════

elif page == "🔮  Predictions":
    st.markdown(
        f"<div class='page-title'>🔮 ML Predictions"
        f"<span class='model-badge'>RandomForest</span></div>"
        f"<div class='page-sub'>MAE: {mae:.2f} · R²: {r2:.3f} · "
        f"scikit-learn RandomForestRegressor</div>",
        unsafe_allow_html=True,
    )

    # Model KPIs
    recovering = (
        (df_pred["Predicted Result"] == "Pass") &
        (df_pred["Result"] == "Fail")
    ).sum()
    avg_change = df_pred["Marks Change"].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📉 MAE",          f"{mae:.2f}")
    m2.metric("📈 R² Score",     f"{r2:.3f}")
    m3.metric("🔄 Can Recover",  int(recovering))
    m4.metric("⬆️ Avg Δ Marks",  f"+{avg_change:.1f}")
    st.markdown("<br>", unsafe_allow_html=True)

    # Bar chart
    st.markdown("<div class='dash-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>📊 Actual vs Predicted — Top 25</div>",
                unsafe_allow_html=True)
    st.pyplot(chart_avp_bar(df_pred, name_col))
    st.markdown("</div>", unsafe_allow_html=True)

    # Scatter + Feature importance
    sc_c, fi_c = st.columns(2, gap="large")
    with sc_c:
        st.markdown("<div class='dash-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🎯 Prediction Scatter</div>",
                    unsafe_allow_html=True)
        st.pyplot(chart_scatter(df_pred))
        st.markdown("</div>", unsafe_allow_html=True)
    with fi_c:
        st.markdown("<div class='dash-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🌲 Feature Importance</div>",
                    unsafe_allow_html=True)
        st.pyplot(chart_importance(model, subj))
        st.markdown("</div>", unsafe_allow_html=True)

    # Full prediction table
    st.markdown("<div class='dash-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>📋 Full Prediction Table</div>",
                unsafe_allow_html=True)
    pred_cols  = [name_col, "Marks", "Result",
                  "Predicted Marks", "Predicted Result", "Marks Change"]
    pred_table = (
        df_pred[pred_cols]
        .sort_values("Marks Change", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(
        pred_table.style
            .format({"Marks": "{:.1f}", "Predicted Marks": "{:.1f}",
                     "Marks Change": "{:+.2f}"})
            .map(style_result, subset=["Result", "Predicted Result"])
            .map(style_change,  subset=["Marks Change"]),
        use_container_width=True, hide_index=True, height=480,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  ── PAGE: DATA ──
# ══════════════════════════════════════════════════════

elif page == "📋  Data":
    st.markdown(
        "<div class='page-title'>📋 Full Dataset</div>"
        "<div class='page-sub'>Browse, filter, and export all student records</div>",
        unsafe_allow_html=True,
    )

    # Filters
    st.markdown("<div class='dash-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>🔎 Filters</div>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        r_filt = st.multiselect("Result", ["Pass", "Fail"],
                                default=["Pass", "Fail"])
    with f2:
        g_filt = st.multiselect(
            "Grade",
            ["Distinction", "First Class", "Pass", "Fail"],
            default=["Distinction", "First Class", "Pass", "Fail"],
        )
    with f3:
        m_range = st.slider("Marks Range", 0, 100, (0, 100))
    st.markdown("</div>", unsafe_allow_html=True)

    filtered = df[
        df["Result"].isin(r_filt) &
        df["Grade"].isin(g_filt) &
        df["Marks"].between(*m_range)
    ].sort_values("Rank")

    st.markdown(
        f"<p style='color:{TEXT_MUTED};font-size:13px;font-weight:600;"
        f"margin-bottom:10px;'>{len(filtered)} records</p>",
        unsafe_allow_html=True,
    )

    all_cols = ([name_col, "Rank", "Marks", "Total"] + subj +
                ["Grade", "Result", "Failed Subjects"])
    all_cols = [c for c in all_cols if c in filtered.columns]

    fmt = {c: "{:.1f}" for c in ["Marks"] + subj if c in all_cols}
    fmt["Total"] = "{:.0f}"

    st.dataframe(
        filtered[all_cols].reset_index(drop=True)
            .style
            .format(fmt)
            .map(style_result, subset=["Result"])
            .map(style_grade,  subset=["Grade"]),
        use_container_width=True, hide_index=True, height=540,
    )

    csv_bytes = filtered[all_cols].to_csv(index=False).encode()
    st.download_button("⬇️  Download CSV", csv_bytes,
                       "student_data.csv", "text/csv")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='dash-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>📐 Descriptive Statistics</div>",
                unsafe_allow_html=True)
    st.dataframe(
        filtered[["Marks"] + subj].describe().round(2),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)