"""
Player Behavior Analysis — Streamlit App
Converted from Jupyter Notebook (Pandas project)
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Player Behavior Analysis",
    page_icon="🎮",
    layout="wide",
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
N_PLAYERS = 10000
RANDOM_SEED = 42

TOXICITY_COLOR_MAP = {
    "Normal": "steelblue",
    "Suspect": "gold",
    "Toxique": "orange",
    "Très Toxique": "red",
}


# ═══════════════════════════════════════════════
# DATA FUNCTIONS
# ═══════════════════════════════════════════════

def generate_sample_data(n: int = N_PLAYERS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a synthetic player statistics DataFrame."""
    np.random.seed(seed)

    df = pd.DataFrame({
        "PlayerID":    [f"P{str(i).zfill(4)}" for i in range(1, n + 1)],
        "Kills":       np.random.randint(0, 40, n),
        "Deaths":      np.random.randint(1, 30, n),   # min 1 to avoid division-by-zero
        "Assists":     np.random.randint(0, 40, n),
        "Wins":        np.random.randint(0, 100, n),
        "Losses":      np.random.randint(0, 100, n),
        "Reports":     np.random.randint(0, 20, n),
        "ChatMessages":np.random.randint(0, 500, n),
        "GameTime":    np.random.randint(10, 500, n),  # hours
    }).set_index("PlayerID")

    # Enrich with account info (region & rank)
    np.random.seed(10)
    df["Region"]       = np.random.choice(["EU", "NA", "ASIA", "SA"], n)
    df["OfficialRank"] = np.random.choice(["Bronze", "Silver", "Gold", "Platinum", "Diamond"], n)

    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived KPIs and classification columns."""
    df = df.copy()

    # ── KPIs ──────────────────────────────────
    df["KDA"]            = (df["Kills"] + df["Assists"]) / df["Deaths"]
    total_games          = df["Wins"] + df["Losses"]
    df["WinRate"]        = (df["Wins"] / total_games.replace(0, np.nan)) * 100
    df["ToxicityScore"]  = df["Reports"] * 2 + df["ChatMessages"] / 50
    df["AvgKillsPerHour"]= df["Kills"] / df["GameTime"]
    df["KillDeathRatio"] = (df["Kills"] / df["Deaths"]).round(2)
    df["TotalGames"]     = df["Wins"] + df["Losses"]

    # ── KDA category ──────────────────────────
    df["KDA_Category"] = pd.cut(
        df["KDA"],
        bins=[0, 1, 2, 3, np.inf],
        labels=["Faible", "Moyen", "Bon", "Excellent"],
    )

    # ── WinRate category ──────────────────────
    df["WinRate_Category"] = pd.cut(
        df["WinRate"],
        bins=[0, 40, 55, 70, 100],
        labels=["Débutant", "Intermédiaire", "Avancé", "Expert"],
    )

    # ── Toxicity label ────────────────────────
    conditions = [
        df["ToxicityScore"] >= 30,
        df["ToxicityScore"] >= 15,
        df["ToxicityScore"] >= 5,
    ]
    choices = ["Très Toxique", "Toxique", "Suspect"]
    df["ToxicityLabel"] = np.select(conditions, choices, default="Normal")

    # ── Player type ───────────────────────────
    df["PlayerType"] = df["GameTime"].apply(
        lambda x: "Hardcore" if x > 300 else ("Regular" if x > 100 else "Casual")
    )

    # ── Global score & rank ───────────────────
    df["GlobalScore"] = (
        df["KDA"] * 30 + df["WinRate"] * 0.4 - df["ToxicityScore"] * 2
    ).round(2)
    df["Rank"] = df["GlobalScore"].rank(ascending=False, method="min").fillna(0).astype(int)

    return df


def handle_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Automatically treat numeric columns with missing values:
    - < 5 % NaN  → drop rows
    - 5–30 % NaN → fill with median
    - > 30 % NaN → drop column
    Returns the cleaned DataFrame and a list of log messages.
    """
    df_clean = df.copy()
    log = []
    n = len(df_clean)

    for col in df_clean.select_dtypes(include="number").columns:
        pct = df_clean[col].isna().sum() / n * 100
        if pct == 0:
            continue
        elif pct < 5:
            df_clean = df_clean.dropna(subset=[col])
            log.append(f"**{col}** ({pct:.1f}% NaN) → rows dropped")
        elif pct <= 30:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            log.append(f"**{col}** ({pct:.1f}% NaN) → filled with median ({median_val:.2f})")
        else:
            df_clean = df_clean.drop(columns=[col])
            log.append(f"**{col}** ({pct:.1f}% NaN) → column dropped (too many NaN)")

    return df_clean, log


def inject_missing_values(df: pd.DataFrame, seed: int = 99) -> pd.DataFrame:
    """Inject random NaN into WinRate, KDA, and ToxicityScore for demo purposes."""
    np.random.seed(seed)
    df_dirty = df.copy()
    n = len(df_dirty)
    idx = df_dirty.index

    df_dirty.loc[idx[np.random.choice([True, False], n, p=[0.05, 0.95])], "WinRate"]       = np.nan
    df_dirty.loc[idx[np.random.choice([True, False], n, p=[0.10, 0.90])], "KDA"]           = np.nan
    df_dirty.loc[idx[np.random.choice([True, False], n, p=[0.60, 0.40])], "ToxicityScore"] = np.nan

    return df_dirty


def load_uploaded_file(uploaded) -> pd.DataFrame:
    """Load a CSV or Excel file uploaded via st.file_uploader."""
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded, index_col=0)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded, index_col=0)
    else:
        raise ValueError("Unsupported file type. Please upload CSV or Excel.")


# ═══════════════════════════════════════════════
# VISUALISATION FUNCTIONS
# ═══════════════════════════════════════════════

def plot_overview(df: pd.DataFrame) -> plt.Figure:
    """6-panel overview figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("Player Behavior — Overview", fontsize=15, fontweight="bold")

    # 1. Player type pie
    counts = df["PlayerType"].value_counts()
    axes[0, 0].pie(counts.values, labels=counts.index,
                   autopct="%1.1f%%",
                   colors=["steelblue", "coral", "mediumseagreen"],
                   startangle=90)
    axes[0, 0].set_title("Player Type Distribution")

    # 2. KDA histogram
    axes[0, 1].hist(df["KDA"].dropna(), bins=25, color="steelblue", edgecolor="white")
    axes[0, 1].axvline(df["KDA"].mean(), color="red", linestyle="--",
                       label=f"Mean: {df['KDA'].mean():.2f}")
    axes[0, 1].set_title("KDA Distribution")
    axes[0, 1].set_xlabel("KDA")
    axes[0, 1].set_ylabel("Players")
    axes[0, 1].legend()

    # 3. WinRate by Rank
    if "OfficialRank" in df.columns:
        wr = df.groupby("OfficialRank")["WinRate"].mean().sort_values(ascending=False)
        axes[0, 2].bar(wr.index, wr.values, color="mediumseagreen", edgecolor="white")
        axes[0, 2].set_title("Avg WinRate by Official Rank")
        axes[0, 2].set_ylabel("WinRate (%)")
        axes[0, 2].set_ylim(0, 100)
    else:
        axes[0, 2].axis("off")

    # 4. Toxicity by player type
    tox = df.groupby("PlayerType")["ToxicityScore"].mean().sort_values(ascending=False)
    axes[1, 0].bar(tox.index, tox.values, color="coral", edgecolor="white")
    axes[1, 0].set_title("Avg Toxicity Score by Player Type")
    axes[1, 0].set_ylabel("ToxicityScore")

    # 5. KDA vs WinRate scatter
    for label, grp in df.groupby("ToxicityLabel"):
        axes[1, 1].scatter(
            grp["KDA"], grp["WinRate"],
            alpha=0.5, s=20,
            label=label,
            color=TOXICITY_COLOR_MAP.get(str(label), "gray"),
        )
    axes[1, 1].set_title("KDA vs WinRate (by Toxicity)")
    axes[1, 1].set_xlabel("KDA")
    axes[1, 1].set_ylabel("WinRate (%)")
    axes[1, 1].legend(fontsize=8)

    # 6. Toxicity label counts (horizontal bar)
    tl = df["ToxicityLabel"].value_counts()
    bar_colors = [TOXICITY_COLOR_MAP.get(k, "gray") for k in tl.index]
    axes[1, 2].barh(tl.index, tl.values, color=bar_colors, edgecolor="white")
    axes[1, 2].set_title("Toxicity Level Distribution")
    axes[1, 2].set_xlabel("Players")

    plt.tight_layout()
    return fig


def plot_groupby_bar(grouped_df: pd.DataFrame, col: str, group_col: str) -> plt.Figure:
    """Simple bar chart for a groupby result."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(grouped_df.index.astype(str), grouped_df[col], color="steelblue", edgecolor="white")
    ax.set_title(f"Avg {col} by {group_col}")
    ax.set_ylabel(col)
    ax.set_xlabel(group_col)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════
# EXPORT HELPER
# ═══════════════════════════════════════════════

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv().encode("utf-8")


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="PlayerStats")
    return buf.getvalue()


# ═══════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════

def main():
    # ── Header ────────────────────────────────
    st.title("🎮 Player Behavior Analysis")
    st.markdown(
        """
        Analyse online player statistics — kills, deaths, assists, win rate,
        toxicity score, and more.  
        Upload your own dataset or generate a 10000-player sample to explore the full pipeline.
        """
    )
    st.divider()

    # ══════════════════════════════════════════
    # SECTION 1 — DATA SOURCE
    # ══════════════════════════════════════════
    st.header("📂 1. Data Source")
    col_upload, col_gen = st.columns([2, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Upload a CSV or Excel file (PlayerID as first column / index)",
            type=["csv", "xlsx", "xls"],
        )

    with col_gen:
        st.markdown("**— or —**")
        if st.button("⚡ Generate Sample Dataset ", use_container_width=True):
            raw = generate_sample_data()
            st.session_state["df_raw"] = raw
            st.session_state["df"] = compute_features(raw)
            st.session_state["source"] = "generated"
            st.success("Sample dataset generated!")

    # Load uploaded file
    if uploaded is not None:
        try:
            raw = load_uploaded_file(uploaded)
            st.session_state["df_raw"] = raw
            st.session_state["df"] = compute_features(raw)
            st.session_state["source"] = f"uploaded: {uploaded.name}"
            st.success(f"File loaded: **{uploaded.name}** — {raw.shape[0]} rows, {raw.shape[1]} columns")
        except Exception as e:
            st.error(f"Could not load file: {e}")

    # Guard: no data yet
    if "df" not in st.session_state:
        st.info("👆 Upload a file or generate a sample dataset to get started.")
        st.stop()

    df: pd.DataFrame = st.session_state["df"]

    # ══════════════════════════════════════════
    # SECTION 2 — RAW DATA PREVIEW
    # ══════════════════════════════════════════
    st.divider()
    st.header("🗂️ 2. Dataset Preview")
    st.caption(f"Source: {st.session_state.get('source', 'unknown')} · {df.shape[0]} players · {df.shape[1]} columns")

    preview_cols = ["Kills", "Deaths", "Assists", "Wins", "Losses",
                    "KDA", "WinRate", "ToxicityScore", "PlayerType", "ToxicityLabel", "GlobalScore", "Rank"]
    available_cols = [c for c in preview_cols if c in df.columns]
    st.dataframe(df[available_cols], use_container_width=True, height=300)

    with st.expander("📊 Quick Statistics"):
        st.dataframe(df[available_cols].describe().T.round(2), use_container_width=True)

    # ══════════════════════════════════════════
    # SECTION 3 — FILTERING
    # ══════════════════════════════════════════
    st.divider()
    st.header("🔍 3. Filtering")

    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        if "Kills" in df.columns:
            kills_min, kills_max = int(df["Kills"].min()), int(df["Kills"].max())
            kills_range = st.slider("Kills range", kills_min, kills_max, (kills_min, kills_max))
        else:
            kills_range = None

    with fc2:
        if "WinRate" in df.columns:
            wr_min = float(df["WinRate"].min()) if df["WinRate"].notna().any() else 0.0
            wr_max = float(df["WinRate"].max()) if df["WinRate"].notna().any() else 100.0
            wr_range = st.slider("WinRate (%) range", 0.0, 100.0, (round(wr_min, 1), round(wr_max, 1)), step=0.5)
        else:
            wr_range = None

    with fc3:
        if "PlayerType" in df.columns:
            player_types = ["All"] + sorted(df["PlayerType"].dropna().unique().tolist())
            sel_type = st.selectbox("Player Type", player_types)
        else:
            sel_type = "All"

        if "ToxicityLabel" in df.columns:
            tox_labels = ["All"] + sorted(df["ToxicityLabel"].dropna().unique().tolist())
            sel_tox = st.selectbox("Toxicity Label", tox_labels)
        else:
            sel_tox = "All"

    # Apply filters
    mask = pd.Series(True, index=df.index)
    if kills_range and "Kills" in df.columns:
        mask &= df["Kills"].between(*kills_range)
    if wr_range and "WinRate" in df.columns:
        mask &= df["WinRate"].between(*wr_range)
    if sel_type != "All" and "PlayerType" in df.columns:
        mask &= df["PlayerType"] == sel_type
    if sel_tox != "All" and "ToxicityLabel" in df.columns:
        mask &= df["ToxicityLabel"] == sel_tox

    df_filtered = df[mask]
    st.caption(f"**{len(df_filtered)}** players match the current filters (out of {len(df)}).")
    st.dataframe(df_filtered[available_cols], use_container_width=True, height=280)

    # ══════════════════════════════════════════
    # SECTION 4 — SORTING
    # ══════════════════════════════════════════
    st.divider()
    st.header("↕️ 4. Sorting & Leaderboard")

    sc1, sc2, sc3 = st.columns([2, 1, 1])
    numeric_cols = df_filtered.select_dtypes(include="number").columns.tolist()

    with sc1:
        sort_col = st.selectbox("Sort by", numeric_cols, index=numeric_cols.index("GlobalScore") if "GlobalScore" in numeric_cols else 0)
    with sc2:
        sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)
    with sc3:
        top_n = st.number_input("Show top N", min_value=5, max_value=200, value=10, step=5)

    df_sorted = df_filtered.sort_values(sort_col, ascending=(sort_order == "Ascending")).head(top_n)
    st.dataframe(df_sorted[available_cols], use_container_width=True, height=300)
    
    # ══════════════════════════════════════════
    # SECTION 5 — MISSING VALUES
    # ══════════════════════════════════════════
    st.divider()
    st.header("🧹 5. Missing Values Handler")

    col_mv1, col_mv2 = st.columns(2)

    with col_mv1:
        if st.button("💉 Inject Missing Values (demo)", use_container_width=True):
            st.session_state["df_dirty"] = inject_missing_values(df)
            st.info("NaN injected into WinRate (5%), KDA (10%), ToxicityScore (60%).")

    with col_mv2:
        if "df_dirty" in st.session_state:
            if st.button("✅ Auto-Clean Missing Values", use_container_width=True):
                cleaned, log = handle_missing_values(st.session_state["df_dirty"])
                st.session_state["df_dirty"] = None
                st.session_state["df"] = compute_features(cleaned.drop(
                    columns=[c for c in cleaned.columns if c in
                             ["KDA","WinRate","ToxicityScore","KDA_Category",
                              "WinRate_Category","ToxicityLabel","PlayerType",
                              "GlobalScore","Rank","AvgKillsPerHour","KillDeathRatio","TotalGames"]
                             ], errors="ignore"
                ))
                df = st.session_state["df"]  # refresh local ref
                st.success(f"Cleaned! {len(df)} rows remain.")
                for msg in log:
                    st.markdown(f"- {msg}")

    # Show current NaN summary
    if "df_dirty" in st.session_state and st.session_state["df_dirty"] is not None:
        df_dirty = st.session_state["df_dirty"]
        missing = (df_dirty.isna().sum() / len(df_dirty) * 100).round(1)
        missing = missing[missing > 0].rename("% Missing")
        st.dataframe(missing.to_frame(), use_container_width=True)
    else:
        missing = (df.isna().sum() / len(df) * 100).round(1)
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values in the current dataset.")
        else:
            st.dataframe(missing.rename("% Missing").to_frame(), use_container_width=True)



    # ══════════════════════════════════════════
    # SECTION 6 — GROUPING & AGGREGATION
    # ══════════════════════════════════════════
    st.divider()
    st.header("📊 6. Grouping & Aggregation")

    cat_cols = [c for c in ["PlayerType", "ToxicityLabel", "KDA_Category",
                             "WinRate_Category", "OfficialRank", "Region"] if c in df.columns]
    agg_num_cols = [c for c in ["Kills", "Deaths", "Assists", "KDA",
                                 "WinRate", "ToxicityScore", "GlobalScore"] if c in df.columns]

    ga1, ga2, ga3 = st.columns(3)
    with ga1:
        group_by = st.selectbox("Group by", cat_cols)
    with ga2:
        agg_metric = st.selectbox("Aggregate metric", agg_num_cols)
    with ga3:
        agg_func = st.selectbox("Aggregation function", ["mean", "median", "sum", "count", "max", "min"])

    grouped = df.groupby(group_by)[agg_metric].agg(agg_func).reset_index()
    grouped.columns = [group_by, f"{agg_func}_{agg_metric}"]
    grouped = grouped.sort_values(f"{agg_func}_{agg_metric}", ascending=False)

    st.dataframe(grouped, use_container_width=True)

    # Full summary table
    with st.expander("📋 Full Summary by Player Type"):
        if "PlayerType" in df.columns:
            summary_cols = {c: ["mean", "max"] for c in agg_num_cols}
            summary = df.groupby("PlayerType").agg(**{
                "Players":        ("Kills", "count"),
                "Avg Kills":      ("Kills", "mean"),
                "Avg KDA":        ("KDA", "mean"),
                "Avg WinRate":    ("WinRate", "mean"),
                "Avg Toxicity":   ("ToxicityScore", "mean"),
                "Total GameTime": ("GameTime", "sum"),
            }).round(2)
            st.dataframe(summary, use_container_width=True)

    
    # ══════════════════════════════════════════
    # SECTION 7 — VISUALISATIONS
    # ══════════════════════════════════════════
    st.divider()
    st.header("📈 7. Visualisations")

    tab_overview, tab_bar, tab_line = st.tabs(["Overview Dashboard", "Grouped Bar Chart", "Line Chart"])

    with tab_overview:
        st.pyplot(plot_overview(df), use_container_width=True)

    with tab_bar:
        st.subheader("Grouped Bar Chart")
        gbc1, gbc2 = st.columns(2)
        with gbc1:
            bar_group = st.selectbox("Group by (bar)", cat_cols, key="bar_group")
        with gbc2:
            bar_metric = st.selectbox("Metric (bar)", agg_num_cols, key="bar_metric")

        gb_result = df.groupby(bar_group)[bar_metric].mean().reset_index()
        gb_result = gb_result.sort_values(bar_metric, ascending=False).set_index(bar_group)
        st.bar_chart(gb_result)

    with tab_line:
        st.subheader("KDA & WinRate — Top 30 Players (by GlobalScore)")
        if "GlobalScore" in df.columns and "KDA" in df.columns and "WinRate" in df.columns:
            top30 = df.sort_values("GlobalScore", ascending=False).head(30)[["KDA", "WinRate"]]
            st.line_chart(top30)
        else:
            st.info("KDA, WinRate or GlobalScore columns not available.")

    # ══════════════════════════════════════════
    # SECTION 8 — EXPORT
    # ══════════════════════════════════════════
    st.divider()
    st.header("💾 8. Export Data")

    exp1, exp2, exp3 = st.columns(3)

    with exp1:
        st.download_button(
            label="⬇️ Download Full Dataset (CSV)",
            data=df_to_csv_bytes(df),
            file_name="player_analysis_full.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with exp2:
        st.download_button(
            label="⬇️ Download Filtered View (CSV)",
            data=df_to_csv_bytes(df_filtered),
            file_name="player_analysis_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with exp3:
        st.download_button(
            label="⬇️ Download Full Dataset (Excel)",
            data=df_to_excel_bytes(df),
            file_name="player_analysis_full.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    # ── Footer ────────────────────────────────
    st.divider()
    st.caption(
        "Player Behavior Analysis App · Built with Streamlit & Pandas · "
    )


if __name__ == "__main__":
    main()
