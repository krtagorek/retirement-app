from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from utils import (
    calculate_compound_growth,
    load_financial_csv,
    run_langgraph_inflation_agent,
    summarize_financial_csv,
    validate_financial_csv,
    synthesize_guidance,
)


load_dotenv()

st.set_page_config(page_title="Retirement Readiness App", layout="wide")


def summarize_dataframe(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return {"rows": 0, "columns": []}
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "null_counts": df.isna().sum().to_dict(),
    }


def build_cashflow_chart(df: pd.DataFrame) -> pd.DataFrame:
    normalized = {column.strip().lower(): column for column in df.columns}
    if "month" not in normalized:
        return pd.DataFrame()

    month_column = normalized["month"]
    chart_columns = [
        ("income", "Income"),
        ("expenses", "Expenses"),
        ("profit", "Profit"),
    ]

    chart_data = pd.DataFrame({"Month": pd.to_datetime(df[month_column])})
    for source_name, display_name in chart_columns:
        if source_name in normalized:
            chart_data[display_name] = pd.to_numeric(df[normalized[source_name]], errors="coerce")

    return chart_data.dropna(subset=["Month"])


def render_data_dictionary() -> None:
    dictionary_rows = [
        ("month", "Month the row represents", "Used to order and chart your last 12 months."),
        ("income", "Gross or take-home monthly income", "Used for income trend and average monthly income."),
        ("expenses", "Total monthly spending", "Used to estimate cashflow pressure and savings potential."),
        ("profit", "Income minus expenses", "Used as a simple monthly surplus/deficit indicator."),
        ("retirement_401k_balance", "Current 401(k) balance in dollars", "Used as the starting balance for projection."),
        ("roth_ira_balance", "Current Roth IRA balance in dollars", "Included in current retirement assets."),
        ("taxable_investments", "Brokerage or other taxable investments", "Included in total assets."),
        ("cash_savings", "Checking/savings balance", "Included in total assets."),
        ("home_equity", "Estimated home equity", "Included in total assets if provided."),
        ("other_assets", "Other assets in dollars", "Optional catch-all asset field."),
        ("total_assets", "All assets combined in dollars", "Displayed as your current asset total."),
        ("notes", "Free-text notes", "Optional context for each month."),
    ]
    st.dataframe(
        pd.DataFrame(dictionary_rows, columns=["Column", "Meaning", "How the app uses it"]),
        use_container_width=True,
        hide_index=True,
    )


def build_projection_chart(projection) -> pd.DataFrame:
    years = max(int(projection.years), 1)
    values = []
    balance = projection.starting_balance
    for year in range(1, years + 1):
        balance = balance * (1 + projection.annual_return_rate) + projection.annual_contribution
        values.append({"Year": year, "Projected Balance": balance})
    return pd.DataFrame(values)


def main() -> None:
    st.title("Retirement Readiness App")
    st.caption("Deterministic math + macro context + LLM guidance")

    if "projection" not in st.session_state:
        st.session_state.projection = None
    if "guidance" not in st.session_state:
        st.session_state.guidance = ""
    if "inflation_context" not in st.session_state:
        st.session_state.inflation_context = {}

    tab_inputs, tab_results = st.tabs(["Inputs & Data Setup", "Projections & LLM Guidance"])

    with tab_inputs:
        st.subheader("1. Upload Financial CSV")
        uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])
        financial_df = load_financial_csv(uploaded_file)
        if not financial_df.empty:
            validation = validate_financial_csv(financial_df)
            if validation.is_valid:
                st.success("CSV loaded successfully.")
            else:
                st.warning("CSV loaded, but validation found issues.")
                for issue in validation.issues:
                    st.write(f"- {issue}")

            st.dataframe(financial_df, use_container_width=True)
            with st.expander("Data dictionary"):
                render_data_dictionary()
        else:
            st.info("Upload a CSV to preview your financial data.")

        st.subheader("2. Input Variables")
        with st.form("retirement_inputs"):
            current_age = st.number_input("Current Age", min_value=0, max_value=120, value=30, step=1)
            target_age = st.number_input("Target Age", min_value=0, max_value=120, value=65, step=1)
            annual_contribution = st.number_input("401(k) Contribution", min_value=0.0, value=10000.0, step=500.0)
            desired_income = st.number_input("Desired Income", min_value=0.0, value=70000.0, step=1000.0)
            submitted = st.form_submit_button("Generate Report")

        if submitted:
            financial_snapshot = summarize_financial_csv(financial_df) if not financial_df.empty else None
            starting_balance = financial_snapshot.current_retirement_balance if financial_snapshot else 0.0
            projection = calculate_compound_growth(
                current_age=int(current_age),
                target_age=int(target_age),
                annual_contribution=float(annual_contribution),
                desired_income=float(desired_income),
                starting_balance=starting_balance,
            )
            st.session_state.projection = projection
            st.session_state.inflation_context = run_langgraph_inflation_agent().get("inflation_context", {})

            dataframe_summary = summarize_dataframe(financial_df)
            if financial_snapshot:
                dataframe_summary["financial_snapshot"] = {
                    "monthly_income_avg": financial_snapshot.monthly_income_avg,
                    "monthly_expenses_avg": financial_snapshot.monthly_expenses_avg,
                    "monthly_profit_avg": financial_snapshot.monthly_profit_avg,
                    "current_retirement_balance": financial_snapshot.current_retirement_balance,
                    "current_asset_total": financial_snapshot.current_asset_total,
                }
            st.session_state.guidance = synthesize_guidance(
                projection=projection,
                inflation_context=st.session_state.inflation_context,
                dataframe_summary=dataframe_summary,
            )
            st.success("Projection and guidance generated. Switch to the next tab.")

    with tab_results:
        st.subheader("Projection Summary")
        projection = st.session_state.projection
        if projection is None:
            st.info("Run the report from Tab 1 to view projections and guidance.")
            return

        projection_data = asdict(projection)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Projected Balance", f"${projection_data['projected_balance']:,.2f}")
        col2.metric("Years to Target", projection_data["years"])
        col3.metric("Annual Return", f"{projection_data['annual_return_rate']:.1%}")
        col4.metric("Income Gap", f"${projection_data['gap_to_target_income']:,.2f}")

        chart_data = build_projection_chart(projection)
        st.subheader("Projected Growth Chart")
        st.line_chart(chart_data.set_index("Year"))

        st.subheader("Deterministic Model Output")
        st.json(projection_data)

        if not financial_df.empty:
            st.subheader("Financial Snapshot")
            snapshot = summarize_financial_csv(financial_df)
            snapshot_cols = st.columns(3)
            snapshot_cols[0].metric("Avg Monthly Income", f"${snapshot.monthly_income_avg:,.2f}")
            snapshot_cols[1].metric("Avg Monthly Expenses", f"${snapshot.monthly_expenses_avg:,.2f}")
            snapshot_cols[2].metric("Avg Monthly Profit", f"${snapshot.monthly_profit_avg:,.2f}")

            asset_cols = st.columns(2)
            asset_cols[0].metric("Current Retirement Balance", f"${snapshot.current_retirement_balance:,.2f}")
            asset_cols[1].metric("Current Asset Total", f"${snapshot.current_asset_total:,.2f}")

            cashflow_chart = build_cashflow_chart(financial_df)
            if not cashflow_chart.empty:
                st.subheader("Last 12 Months Cash Flow")
                st.line_chart(cashflow_chart.set_index("Month"))

        st.subheader("Inflation Context")
        st.json(st.session_state.inflation_context)

        st.subheader("LLM Guidance")
        st.write(st.session_state.guidance or "No guidance generated yet.")


if __name__ == "__main__":
    main()
