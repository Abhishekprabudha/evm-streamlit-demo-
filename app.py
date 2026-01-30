import re
from io import BytesIO
from datetime import datetime, date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="EVM Demo (PV / EV / AC)", layout="wide")

EXPECTED_SHEETS = {
    "plan": ["Project_Plan", "Project Plan", "Plan"],
    "progress": ["Progress_Updates", "Progress Updates", "Updates", "Progress"],
}

PLAN_REQUIRED_COLS = {
    "Task ID": ["Task ID", "WBS", "ID"],
    "Task Name": ["Task Name", "Name", "Activity"],
    "Type": ["Type", "Task Type"],
    "Start": ["Start", "Start Date"],
    "Finish": ["Finish", "Finish Date", "End", "End Date"],
    "Duration (days)": ["Duration (days)", "Duration", "Duration Days"],
    "Baseline Cost (BAC)": ["Baseline Cost (BAC)", "BAC", "Budget", "Baseline Cost"],
}

PROG_REQUIRED_COLS = {
    "Week Ending": ["Week Ending", "As Of", "Week"],
    "Task ID": ["Task ID", "WBS", "ID"],
    "% Complete (as of week)": ["% Complete (as of week)", "% Complete", "Percent Complete", "Progress %"],
    "Actual Cost to Date": ["Actual Cost to Date", "AC", "Actual Cost", "Cost Actual"],
}

DEFAULT_RISK_LOOKAHEAD_DAYS = 14


# ----------------------------
# Helpers
# ----------------------------
def _find_sheet_name(xls: pd.ExcelFile, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in xls.sheet_names:
            return c
    # fallback: fuzzy contains
    for name in xls.sheet_names:
        low = name.lower().replace(" ", "_")
        for c in candidates:
            if c.lower().replace(" ", "_") in low:
                return name
    return None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _map_columns(df: pd.DataFrame, required_map: dict[str, list[str]]) -> pd.DataFrame:
    df = _normalize_columns(df)
    col_map = {}

    lower_cols = {c.lower(): c for c in df.columns}
    for canonical, aliases in required_map.items():
        found = None
        for a in aliases:
            key = a.lower()
            if key in lower_cols:
                found = lower_cols[key]
                break
        if not found:
            # try contains
            for c in df.columns:
                if key in c.lower():
                    found = c
                    break
        if not found:
            raise ValueError(f"Missing required column: '{canonical}'. Tried aliases: {aliases}")
        col_map[found] = canonical

    df = df.rename(columns=col_map)
    return df[list(required_map.keys())]


def _to_date_series(s: pd.Series) -> pd.Series:
    # Works for Excel serials, strings, datetimes
    out = pd.to_datetime(s, errors="coerce").dt.date
    if out.isna().any():
        # try Excel numeric date
        out2 = pd.to_datetime(s, errors="coerce", unit="D", origin="1899-12-30").dt.date
        out = out.fillna(out2)
    if out.isna().any():
        raise ValueError("Could not parse some dates. Ensure date columns are valid Excel dates or ISO strings.")
    return out


def build_timephased_pv(plan_df: pd.DataFrame, asof_dates: list[date]) -> pd.DataFrame:
    """
    PV(t) computed by linearly spreading each task's BAC across its planned duration.
    Milestones (duration 0 or BAC 0) contribute 0.
    """
    df = plan_df.copy()
    df["Start"] = _to_date_series(df["Start"])
    df["Finish"] = _to_date_series(df["Finish"])

    # duration safety
    df["Duration (days)"] = pd.to_numeric(df["Duration (days)"], errors="coerce").fillna(0).astype(int)
    df["Baseline Cost (BAC)"] = pd.to_numeric(df["Baseline Cost (BAC)"], errors="coerce").fillna(0).astype(float)

    # treat milestones as 0 PV (as in your template)
    is_task = (df["Baseline Cost (BAC)"] > 0) & (df["Duration (days)"] > 0)

    rows = []
    for d in asof_dates:
        pv_total = 0.0
        for _, r in df[is_task].iterrows():
            start, finish, dur, bac = r["Start"], r["Finish"], r["Duration (days)"], r["Baseline Cost (BAC)"]
            if d < start:
                pv = 0.0
            elif d >= finish:
                pv = float(bac)
            else:
                elapsed = (d - start).days
                pv = float(bac) * max(0.0, min(1.0, elapsed / max(dur, 1)))
            pv_total += pv
        rows.append({"As Of": d, "PV": pv_total})
    return pd.DataFrame(rows)


def build_ev_ac(plan_df: pd.DataFrame, prog_df: pd.DataFrame) -> pd.DataFrame:
    """
    EV per task = BAC * %complete.
    AC per task = actual cost to date from progress sheet.
    Aggregated by week ending.
    """
    plan = plan_df.copy()
    plan["Baseline Cost (BAC)"] = pd.to_numeric(plan["Baseline Cost (BAC)"], errors="coerce").fillna(0).astype(float)

    prog = prog_df.copy()
    prog["Week Ending"] = _to_date_series(prog["Week Ending"])
    prog["% Complete (as of week)"] = pd.to_numeric(prog["% Complete (as of week)"], errors="coerce").fillna(0).astype(float)
    prog["Actual Cost to Date"] = pd.to_numeric(prog["Actual Cost to Date"], errors="coerce").fillna(0).astype(float)

    merged = prog.merge(plan[["Task ID", "Task Name", "Type", "Baseline Cost (BAC)"]], on="Task ID", how="left")
    merged["Baseline Cost (BAC)"] = merged["Baseline Cost (BAC)"].fillna(0.0)

    merged["EV"] = merged["Baseline Cost (BAC)"] * (merged["% Complete (as of week)"] / 100.0)
    merged["AC"] = merged["Actual Cost to Date"]

    agg = (
        merged.groupby("Week Ending", as_index=False)[["EV", "AC"]]
        .sum()
        .rename(columns={"Week Ending": "As Of"})
        .sort_values("As Of")
    )
    return agg, merged


def add_spi_cpi(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SPI"] = out.apply(lambda r: (r["EV"] / r["PV"]) if r["PV"] > 0 else None, axis=1)
    out["CPI"] = out.apply(lambda r: (r["EV"] / r["AC"]) if r["AC"] > 0 else None, axis=1)
    return out


def milestone_risk(plan_df: pd.DataFrame, merged_task_progress: pd.DataFrame, asof: date, lookahead_days: int):
    """
    Risk logic (demo-friendly):
    - Milestone at risk if:
      A) As-of date > milestone planned date AND milestone not 100% complete
      OR
      B) Milestone planned date within lookahead window AND
         predecessor chain tasks show schedule slippage (avg SPI of predecessors < 1.0)
    Mitigation: simple playbook based on phase/type.
    """
    plan = plan_df.copy()
    plan["Start"] = _to_date_series(plan["Start"])
    plan["Finish"] = _to_date_series(plan["Finish"])

    # milestones = Type contains 'milestone' OR duration == 0
    is_ms = plan["Type"].astype(str).str.lower().str.contains("milestone") | (pd.to_numeric(plan["Duration (days)"], errors="coerce").fillna(0) == 0)
    milestones = plan[is_ms].copy()

    # latest progress snapshot at/before asof
    snap = merged_task_progress.copy()
    snap = snap[snap["Week Ending"] <= asof].sort_values(["Task ID", "Week Ending"])
    latest = snap.groupby("Task ID", as_index=False).tail(1)[["Task ID", "% Complete (as of week)", "EV", "AC", "Baseline Cost (BAC)"]]

    # compute a simple "expected % complete by asof" for tasks using planned dates
    tasks_only = plan[~is_ms].copy()
    tasks_only["Duration (days)"] = pd.to_numeric(tasks_only["Duration (days)"], errors="coerce").fillna(0).astype(int)
    def expected_pct(row):
        if asof < row["Start"]:
            return 0.0
        if asof >= row["Finish"]:
            return 100.0
        elapsed = (asof - row["Start"]).days
        return 100.0 * max(0.0, min(1.0, elapsed / max(row["Duration (days)"], 1)))
    tasks_only["Expected % (plan)"] = tasks_only.apply(expected_pct, axis=1)

    # join expected vs actual
    task_status = tasks_only.merge(latest, on="Task ID", how="left")
    task_status["Actual %"] = task_status["% Complete (as of week)"].fillna(0.0)

    # task SPI proxy at snapshot: EV/PV_task(asof) (local)
    # PV_task(asof) same linear spread
    def pv_task_asof(row):
        bac = float(row["Baseline Cost (BAC)"] or 0.0)
        if bac <= 0:
            return 0.0
        if asof < row["Start"]:
            return 0.0
        if asof >= row["Finish"]:
            return bac
        elapsed = (asof - row["Start"]).days
        return bac * max(0.0, min(1.0, elapsed / max(int(row["Duration (days)"]), 1)))
    task_status["PV_task"] = task_status.apply(pv_task_asof, axis=1)
    task_status["SPI_task"] = task_status.apply(lambda r: (r["EV"] / r["PV_task"]) if (r["PV_task"] and r["PV_task"] > 0) else None, axis=1)

    # milestone completion from progress (if present)
    ms_latest = latest.rename(columns={"% Complete (as of week)": "Milestone %"})
    milestones = milestones.merge(ms_latest[["Task ID", "Milestone %"]], on="Task ID", how="left")
    milestones["Milestone %"] = milestones["Milestone %"].fillna(0.0)

    lookahead_end = asof + pd.Timedelta(days=lookahead_days)
    results = []

    for _, ms in milestones.iterrows():
        ms_date = ms["Start"]  # milestone planned date
        overdue = (asof > ms_date) and (ms["Milestone %"] < 100.0)
        upcoming = (asof <= ms_date <= lookahead_end.date())

        # predecessor SPI estimate: if predecessor field exists, evaluate that predecessor task SPI
        preds = str(ms.get("Predecessor") or "").strip()
        pred_ids = [p.strip() for p in re.split(r"[;,]", preds) if p.strip()]
        pred_spi_vals = []
        for pid in pred_ids:
            row = task_status[task_status["Task ID"] == pid]
            if not row.empty:
                v = row.iloc[0]["SPI_task"]
                if pd.notna(v):
                    pred_spi_vals.append(float(v))

        pred_spi_avg = sum(pred_spi_vals) / len(pred_spi_vals) if pred_spi_vals else None
        pred_risk = upcoming and (pred_spi_avg is not None) and (pred_spi_avg < 1.0)

        if overdue or pred_risk:
            severity = "High" if overdue else "Medium"
            reason = []
            if overdue:
                reason.append(f"Overdue (planned {ms_date.isoformat()}, completion {ms['Milestone %']:.0f}%).")
            if pred_risk:
                reason.append(f"Predecessor SPI below 1.0 (avg {pred_spi_avg:.2f}) for upcoming milestone.")
            reason = " ".join(reason)

            mitigation = suggest_mitigation(ms["Phase"], severity)

            results.append({
                "Milestone ID": ms["Task ID"],
                "Milestone": ms["Task Name"],
                "Phase": ms["Phase"],
                "Planned Date": ms_date,
                "Completion %": ms["Milestone %"],
                "Risk": severity,
                "Reason": reason,
                "Mitigation": mitigation,
            })

    out = pd.DataFrame(results).sort_values(["Risk", "Planned Date"], ascending=[True, True]) if results else pd.DataFrame(
        columns=["Milestone ID", "Milestone", "Phase", "Planned Date", "Completion %", "Risk", "Reason", "Mitigation"]
    )
    return out


def suggest_mitigation(phase: str, severity: str) -> str:
    base = []
    if severity == "High":
        base.append("Stand up a 48–72 hour recovery plan with daily checkpoints.")
        base.append("Re-baseline critical path tasks; enforce WIP limits and remove blockers fast.")
    else:
        base.append("Tighten weekly plan; add mid-week checkpoint and re-sequence work to protect critical path.")

    phase_low = (phase or "").lower()
    if "init" in phase_low:
        base.append("Escalate decision owners; lock scope assumptions and approval SLAs.")
    elif "design" in phase_low:
        base.append("Timebox design reviews; approve with 'fit-for-build' criteria; defer non-critical UI polish.")
    elif "build" in phase_low:
        base.append("Add a short-term strike team; parallelize modules; focus on highest EV features first.")
    elif "test" in phase_low:
        base.append("Introduce risk-based test suite; automate smoke; isolate flaky environments; fix top defect clusters first.")
    elif "deploy" in phase_low:
        base.append("Freeze change window; pre-stage releases; run rehearsal + rollback drill; increase hypercare staffing.")
    else:
        base.append("Reconfirm dependencies and resource coverage; add explicit owner + due-date per blocker.")

    return " ".join(base)


def fig_evm_lines(evm_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=evm_df["As Of"], y=evm_df["PV"], mode="lines+markers", name="PV"))
    fig.add_trace(go.Scatter(x=evm_df["As Of"], y=evm_df["EV"], mode="lines+markers", name="EV"))
    fig.add_trace(go.Scatter(x=evm_df["As Of"], y=evm_df["AC"], mode="lines+markers", name="AC"))
    fig.update_layout(
        title="Cumulative PV / EV / AC",
        xaxis_title="As-of Date",
        yaxis_title="Value",
        hovermode="x unified",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def fig_spi_cpi(evm_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=evm_df["As Of"], y=evm_df["SPI"], mode="lines+markers", name="SPI"))
    fig.add_trace(go.Scatter(x=evm_df["As Of"], y=evm_df["CPI"], mode="lines+markers", name="CPI"))
    fig.add_hline(y=1.0, line_dash="dash")
    fig.update_layout(
        title="Performance Indices (SPI / CPI)",
        xaxis_title="As-of Date",
        yaxis_title="Index (1.0 = on plan)",
        hovermode="x unified",
        height=320,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def to_excel_bytes(evm_df: pd.DataFrame, milestone_df: pd.DataFrame, task_snapshot_df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        evm_df.to_excel(writer, index=False, sheet_name="EVM_TimeSeries")
        milestone_df.to_excel(writer, index=False, sheet_name="Milestone_Risk")
        task_snapshot_df.to_excel(writer, index=False, sheet_name="Task_Snapshot")
    return output.getvalue()


# ----------------------------
# UI: Sidebar controls
# ----------------------------
st.title("Earned Value Management (EVM) Demo — PV / EV / AC + SPI / CPI")

with st.sidebar:
    st.header("Demo Controls")
    refresh_minutes = st.slider("Auto-refresh (minutes)", min_value=0, max_value=30, value=0, step=1)
    lookahead_days = st.slider("Milestone risk lookahead (days)", 7, 45, DEFAULT_RISK_LOOKAHEAD_DAYS, 1)

    st.markdown("---")
    uploaded = st.file_uploader("Upload Project Plan (.xlsx)", type=["xlsx"])

if refresh_minutes and refresh_minutes > 0:
    st_autorefresh(interval=refresh_minutes * 60 * 1000, key="evm_autorefresh")


if not uploaded:
    st.info("Upload your Excel project plan to compute EVM metrics and render dashboards.")
    st.stop()

# ----------------------------
# Load workbook
# ----------------------------
try:
    xls = pd.ExcelFile(uploaded)
    plan_sheet = _find_sheet_name(xls, EXPECTED_SHEETS["plan"])
    prog_sheet = _find_sheet_name(xls, EXPECTED_SHEETS["progress"])

    if not plan_sheet or not prog_sheet:
        raise ValueError(
            f"Could not find required sheets.\n"
            f"Found sheets: {xls.sheet_names}\n"
            f"Expected something like: {EXPECTED_SHEETS['plan']} and {EXPECTED_SHEETS['progress']}"
        )

    raw_plan = pd.read_excel(xls, sheet_name=plan_sheet)
    raw_prog = pd.read_excel(xls, sheet_name=prog_sheet)

    plan_df = _map_columns(raw_plan, PLAN_REQUIRED_COLS)
    prog_df = _map_columns(raw_prog, PROG_REQUIRED_COLS)

except Exception as e:
    st.error(f"Failed to read workbook: {e}")
    st.stop()

# As-of dates from progress sheet (weekly)
asof_dates = sorted(pd.to_datetime(prog_df["Week Ending"], errors="coerce").dt.date.unique().tolist())
if not asof_dates:
    st.error("No valid 'Week Ending' dates found in Progress_Updates.")
    st.stop()

# As-of selector
colA, colB, colC = st.columns([1.2, 1.2, 2])
with colA:
    asof = st.selectbox("As-of Date (reporting point)", asof_dates, index=len(asof_dates) - 1)
with colB:
    currency = st.selectbox("Currency", ["USD", "AED", "INR", "EUR", "GBP"], index=0)
with colC:
    st.caption(f"Loaded sheets: **{plan_sheet}**, **{prog_sheet}**")

# ----------------------------
# Compute EVM
# ----------------------------
pv_df = build_timephased_pv(plan_df, asof_dates)
evac_df, merged_task_progress = build_ev_ac(plan_df, prog_df)
evm_df = pv_df.merge(evac_df, on="As Of", how="left").fillna({"EV": 0.0, "AC": 0.0}).sort_values("As Of")
evm_df = add_spi_cpi(evm_df)

# Project snapshot
snap_row = evm_df[evm_df["As Of"] == asof].iloc[0]
PV, EV, AC = float(snap_row["PV"]), float(snap_row["EV"]), float(snap_row["AC"])
SPI, CPI = snap_row["SPI"], snap_row["CPI"]

# Milestone risk view
milestones_at_risk = milestone_risk(plan_df, merged_task_progress, asof=asof, lookahead_days=lookahead_days)

# Task snapshot at asof
snap = merged_task_progress.copy()
snap = snap[snap["Week Ending"] <= asof].sort_values(["Task ID", "Week Ending"])
task_latest = snap.groupby("Task ID", as_index=False).tail(1)[
    ["Task ID", "Task Name", "Type", "% Complete (as of week)", "EV", "AC", "Baseline Cost (BAC)"]
].rename(columns={"% Complete (as of week)": "% Complete"})

# ----------------------------
# Dashboard
# ----------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("PV", f"{PV:,.0f} {currency}")
k2.metric("EV", f"{EV:,.0f} {currency}")
k3.metric("AC", f"{AC:,.0f} {currency}")
k4.metric("SPI", f"{SPI:.2f}" if SPI is not None else "—")
k5.metric("CPI", f"{CPI:.2f}" if CPI is not None else "—")

tab1, tab2, tab3 = st.tabs(["Project Dashboard", "Task / Milestone Views", "Ask: Milestone Risk & Mitigation"])

with tab1:
    left, right = st.columns([2, 1])

    with left:
        st.plotly_chart(fig_evm_lines(evm_df), use_container_width=True)
    with right:
        st.plotly_chart(fig_spi_cpi(evm_df), use_container_width=True)

        st.markdown("#### Interpretation (demo-friendly)")
        st.write(
            "- **SPI < 1.0** → behind schedule; **SPI > 1.0** → ahead.\n"
            "- **CPI < 1.0** → over budget; **CPI > 1.0** → under budget.\n"
            "- Values are **cumulative** and derived from baseline + progress snapshots."
        )

    st.markdown("#### Download Reports")
    csv_bytes = evm_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download EVM Timeseries (CSV)", data=csv_bytes, file_name="evm_timeseries.csv", mime="text/csv")

    excel_bytes = to_excel_bytes(evm_df, milestones_at_risk, task_latest)
    st.download_button(
        "Download Full Report (Excel)",
        data=excel_bytes,
        file_name="evm_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

with tab2:
    st.subheader("Task-level snapshot (latest up to as-of)")
    st.dataframe(task_latest.sort_values(["Type", "Task ID"]), use_container_width=True, height=420)

    st.subheader("Milestones at risk")
    if milestones_at_risk.empty:
        st.success("No milestones flagged as at-risk for the selected window.")
    else:
        st.dataframe(milestones_at_risk, use_container_width=True, height=320)

with tab3:
    st.subheader("Ask about milestone risk + mitigation")
    st.caption("This is a deterministic demo (no external AI needed). It maps your question to risk logic + a mitigation playbook.")

    q = st.text_input("Ask a question", value="Which milestones are at risk and what mitigation do you recommend?")
    ask_btn = st.button("Answer")

    if ask_btn:
        q_low = q.strip().lower()

        if ("milestone" in q_low and ("risk" in q_low or "at risk" in q_low or "delayed" in q_low)) or q_low.startswith("which"):
            if milestones_at_risk.empty:
                st.write("✅ No milestones are currently flagged as at-risk within the configured lookahead window.")
            else:
                st.write(f"⚠️ **{len(milestones_at_risk)} milestone(s)** flagged as at-risk as of **{asof.isoformat()}**:")
                for _, r in milestones_at_risk.iterrows():
                    st.markdown(
                        f"- **{r['Milestone ID']} — {r['Milestone']}** (Risk: **{r['Risk']}**, Planned: {r['Planned Date']})\n"
                        f"  - Reason: {r['Reason']}\n"
                        f"  - Mitigation: {r['Mitigation']}"
                    )
        elif "spi" in q_low or "cpi" in q_low:
            st.write(f"As of **{asof.isoformat()}**: SPI = **{SPI:.2f}** and CPI = **{CPI:.2f}**." if (SPI is not None and CPI is not None) else "SPI/CPI not available for this point (PV or AC was zero).")
        else:
            st.write(
                "Try questions like:\n"
                "- “Which milestones are at risk?”\n"
                "- “What mitigation should we take for at-risk milestones?”\n"
                "- “What are SPI and CPI saying?”"
            )

st.caption("EVM definitions used: PV (planned time-phased budget), EV (BAC × %complete), AC (actual cost). SPI=EV/PV, CPI=EV/AC. :contentReference[oaicite:1]{index=1}")
