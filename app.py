from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from utils import (
    calculate_compound_growth,
    load_financial_csv,
    run_langgraph_inflation_agent,
    run_what_if_analysis,
    summarize_financial_csv,
    validate_financial_csv,
    synthesize_guidance,
)


load_dotenv()

st.set_page_config(page_title="Retirement Readiness Check", layout="wide")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
            }
            .hero-card {
                background: linear-gradient(135deg, #4f46e5 0%, #2563eb 55%, #0ea5e9 100%);
                color: white;
                border-radius: 28px;
                padding: 1.4rem 1.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
                overflow: hidden;
                position: relative;
            }
            .hero-card::after {
                content: '';
                position: absolute;
                inset: 0;
                background: radial-gradient(circle at top right, rgba(255,255,255,0.22), transparent 38%);
                pointer-events: none;
            }
            .hero-title {
                font-size: 1.6rem;
                font-weight: 800;
                margin-bottom: 0.25rem;
            }
            .hero-subtitle {
                font-size: 0.95rem;
                opacity: 0.95;
                line-height: 1.5;
            }
            .block-container {
                padding-top: 1.2rem;
                padding-bottom: 2rem;
            }
            .page-shell {
                background: rgba(255, 255, 255, 0.78);
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 24px;
                padding: 1.25rem 1.25rem 0.75rem 1.25rem;
                box-shadow: 0 24px 60px rgba(15, 23, 42, 0.08);
            }
            .stButton > button {
                border-radius: 999px;
                border: 1px solid rgba(79, 70, 229, 0.2);
                background: linear-gradient(135deg, #4f46e5, #2563eb);
                color: white;
                font-weight: 700;
                box-shadow: 0 10px 20px rgba(37, 99, 235, 0.18);
                padding: 0.55rem 1rem;
            }
            .stButton > button:hover {
                border-color: rgba(79, 70, 229, 0.35);
                box-shadow: 0 12px 24px rgba(37, 99, 235, 0.22);
            }
            .summary-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 0.9rem;
            }
            .summary-panel {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 18px;
                padding: 0.85rem 0.9rem 0.75rem 0.9rem;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
                min-height: 100%;
            }
            .section-label {
                font-size: 0.8rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #64748b;
                margin-bottom: 0.35rem;
            }
            .panel-title {
                font-size: 0.95rem;
                font-weight: 800;
                color: #0f172a;
                margin-bottom: 0.15rem;
            }
            .panel-subtitle {
                font-size: 0.85rem;
                color: #64748b;
                margin-bottom: 0.85rem;
            }
            div[data-testid="stDataFrame"] {
                border-radius: 12px;
                overflow: hidden;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    chart_data = pd.DataFrame({"Month": pd.to_datetime(df[month_column])})
    for source_name, display_name in [("income", "Income"), ("expenses", "Expenses"), ("profit", "Profit")]:
        if source_name in normalized:
            chart_data[display_name] = pd.to_numeric(df[normalized[source_name]], errors="coerce")

    return chart_data.dropna(subset=["Month"])


def render_data_dictionary() -> None:
    dictionary_rows = [
        ("month", "Month", "Keeps your 12 months in order for charts."),
        ("income", "Income", "Used to show average monthly income and income trends."),
        ("expenses", "Expenses", "Used to show spending trends and savings room."),
        ("profit", "Profit", "Income minus expenses for each month."),
        ("retirement_401k_balance", "401(k) Balance", "Used as part of your starting retirement balance."),
        ("roth_ira_balance", "Roth IRA Balance", "Included in your retirement assets."),
        ("taxable_investments", "Taxable Investments", "Included in your total assets."),
        ("cash_savings", "Cash Savings", "Included in your total assets."),
        ("home_equity", "Home Equity", "Included in your total assets if provided."),
        ("other_assets", "Other Assets", "Optional extra assets in dollars."),
        ("total_assets", "Total Assets", "Shown as your current asset total."),
        ("notes", "Notes", "Optional monthly context."),
    ]
    st.dataframe(
        pd.DataFrame(dictionary_rows, columns=["Column", "Meaning", "How the app uses it"]),
        use_container_width=True,
        hide_index=True,
    )


def render_data_dictionary_preview(df: pd.DataFrame) -> None:
    if df.empty:
        return

    preview_rows = pd.DataFrame(
        {
            "Column": df.columns,
            "Example": [str(df.iloc[0][column]) for column in df.columns],
        }
    )
    st.caption("Quick preview of the first row in your file.")
    st.dataframe(preview_rows, use_container_width=True, hide_index=True)


def get_sample_template() -> pd.DataFrame:
    return pd.read_csv("sample_financial_data.csv")


def build_projection_chart(projection) -> pd.DataFrame:
    values = []
    balance = projection.starting_balance
    for year in range(1, max(int(projection.years), 1) + 1):
        balance = balance * (1 + projection.annual_return_rate) + projection.annual_contribution
        values.append({"Year": year, "Projected Balance": balance})
    return pd.DataFrame(values)


def render_metric_card(title: str, value: str, help_text: str) -> None:
    st.markdown(
        f"""
        <div style="padding: 1rem 1rem 0.9rem 1rem; border-radius: 1rem; border: 1px solid rgba(148, 163, 184, 0.18); background: rgba(255,255,255,0.95); box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);">
            <div style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; color: #64748b; margin-bottom: 0.35rem;">{title}</div>
            <div style="font-size: 1.55rem; font-weight: 800; color: #0f172a; line-height: 1.1;">{value}</div>
            <div style="font-size: 0.82rem; color: #64748b; margin-top: 0.35rem; line-height: 1.35;">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_grid(title_value_help: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(title_value_help))
    for col, (title, value, help_text) in zip(cols, title_value_help):
        with col:
            render_metric_card(title, value, help_text)


def render_section_intro(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="section-label">{title}</div>', unsafe_allow_html=True)
    st.markdown(f"**{subtitle}**")


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Retirement Readiness Check</div>
            <div class="hero-subtitle">
                Upload your financial history, enter your goals, and get a clear retirement snapshot with
                projection math, inflation context, and friendly guidance.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_card(is_on_track: bool, gap_amount: float) -> None:
    if is_on_track:
        background = "#dcfce7"
        border = "#16a34a"
        title = "Great news — you’re on track"
        body = "Your projected balance appears to support your income goal."
    else:
        background = "#fee2e2"
        border = "#dc2626"
        title = "You’re close — here’s the gap"
        body = f"You may need about ${gap_amount:,.2f} more to better support your income goal."

    st.markdown(
        f"""
        <div style="padding: 1rem 1.1rem; border-radius: 1rem; border: 1px solid {border}; background: {background};">
            <div style="font-size: 1rem; font-weight: 700; color: #0f172a; margin-bottom: 0.25rem;">{title}</div>
            <div style="font-size: 0.92rem; color: #0f172a;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_takeaway_box(is_on_track: bool, gap_amount: float) -> None:
    if is_on_track:
        message = "What this means: your current plan appears to be in a good place. You can use the details below to stay on course."
    else:
        message = f"What this means: you are close, but you may want to close about ${gap_amount:,.2f} of projected gap over time."
    st.info(message)


def render_assumptions_box(projection, inflation_context: dict[str, object]) -> None:
    inflation_source = inflation_context.get("series_id", "FRED CPI") if isinstance(inflation_context, dict) else "FRED CPI"
    st.markdown(
        f"""
        <div style="padding: 1rem 1.1rem; border-radius: 1rem; border: 1px solid #cbd5e1; background: #f8fafc;">
            <div style="font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-bottom: 0.4rem;">Assumptions</div>
            <div style="font-size: 0.9rem; color: #0f172a; line-height: 1.6;">
                Return rate: {projection.annual_return_rate:.1%}<br/>
                Starting retirement balance: ${projection.starting_balance:,.2f}<br/>
                Inflation source: {inflation_source}<br/>
                For educational use only, not financial advice.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_compact_summary_header(projection, inflation_context: dict[str, object]) -> None:
    inflation_source = inflation_context.get("series_id", "FRED CPI") if isinstance(inflation_context, dict) else "FRED CPI"
    st.markdown(
        f"""
        <div style="padding: 1rem 1.1rem; border-radius: 1rem; border: 1px solid rgba(148,163,184,0.18); background: rgba(255,255,255,0.92); margin-top: 0.25rem; margin-bottom: 0.9rem;">
            <div style="font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-bottom: 0.25rem;">Assumptions</div>
            <div style="font-size: 0.86rem; color: #475569; line-height: 1.5;">
                Return rate: {projection.annual_return_rate:.1%} &nbsp;•&nbsp;
                Starting balance: ${projection.starting_balance:,.2f} &nbsp;•&nbsp;
                Inflation source: {inflation_source} &nbsp;•&nbsp;
                Educational use only, not financial advice.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_at_a_glance(projection, inflation_context: dict[str, object]) -> None:
    is_on_track = projection.projected_balance >= projection.desired_income
    status_text = "On track" if is_on_track else "Gap to close"
    status_color = "#16a34a" if is_on_track else "#dc2626"
    inflation_source = inflation_context.get("series_id", "FRED CPI") if isinstance(inflation_context, dict) else "FRED CPI"

    st.markdown(
        f"""
        <div style="display:grid; grid-template-columns: 1.2fr 1fr; gap: 0.8rem; margin-top: 0.15rem; margin-bottom: 0.8rem;">
            <div style="padding: 0.95rem 1rem; border-radius: 1rem; border: 1px solid rgba(148,163,184,0.18); background: rgba(255,255,255,0.94); box-shadow: 0 10px 24px rgba(15,23,42,0.05);">
                <div style="font-size: 0.8rem; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase; color: #64748b; margin-bottom: 0.35rem;">Main takeaway</div>
                <div style="font-size: 1.2rem; font-weight: 800; color: {status_color}; margin-bottom: 0.2rem;">{status_text}</div>
                <div style="font-size: 0.9rem; color: #334155; line-height: 1.45;">
                    Return rate {projection.annual_return_rate:.1%} • Starting balance ${projection.starting_balance:,.2f}
                </div>
            </div>
            <div style="padding: 0.95rem 1rem; border-radius: 1rem; border: 1px solid rgba(148,163,184,0.18); background: rgba(255,255,255,0.94); box-shadow: 0 10px 24px rgba(15,23,42,0.05);">
                <div style="font-size: 0.8rem; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase; color: #64748b; margin-bottom: 0.35rem;">Assumptions</div>
                <div style="font-size: 0.86rem; color: #334155; line-height: 1.55;">
                    Inflation source: {inflation_source}<br/>
                    Educational use only, not financial advice.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_scenario_summary(updated_inputs: dict[str, object], updated_projection) -> None:
    st.markdown(
        f"""
        <div style="padding: 0.95rem 1rem; border-radius: 1rem; border: 1px solid rgba(148,163,184,0.18); background: rgba(255,255,255,0.95); box-shadow: 0 10px 24px rgba(15,23,42,0.05); margin-bottom: 0.6rem;">
            <div style="font-size: 0.8rem; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase; color: #64748b; margin-bottom: 0.3rem;">Updated scenario</div>
            <div style="font-size: 0.9rem; color: #334155; line-height: 1.45;">
                Current age: {updated_inputs.get('current_age')} • Target age: {updated_inputs.get('target_age')} • 
                Annual contribution: ${float(updated_inputs.get('annual_contribution', 0.0)):,.2f} • 
                Projected balance: ${updated_projection.projected_balance:,.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_overview(projection, financial_df: pd.DataFrame, inflation_context: dict[str, object]) -> None:
    projection_data = asdict(projection)
    is_on_track = projection_data["gap_to_target_income"] <= 0
    render_status_card(is_on_track, projection_data["gap_to_target_income"])
    render_summary_at_a_glance(projection, inflation_context)

    render_metric_grid([
        ("Projected Balance", f"${projection_data['projected_balance']:,.2f}", "Estimated balance at your target age"),
        ("Years to Target", str(projection_data["years"]), "Time remaining in the projection"),
        ("Annual Return", f"{projection_data['annual_return_rate']:.1%}", "Assumed long-term growth rate"),
        ("Income Gap", f"${projection_data['gap_to_target_income']:,.2f}", "Difference versus desired income"),
    ])

    col_chart, col_snapshot = st.columns([1.2, 1])
    with col_chart:
        chart_data = build_projection_chart(projection)
        st.markdown("**Projected Growth Chart**")
        st.line_chart(chart_data.set_index("Year"))

    with col_snapshot:
        if not financial_df.empty:
            st.markdown("**Your Financial Snapshot**")
            snapshot = summarize_financial_csv(financial_df)
            render_metric_grid([
                ("Avg Monthly Income", f"${snapshot.monthly_income_avg:,.2f}", "Average monthly income over the last 12 months"),
                ("Avg Monthly Expenses", f"${snapshot.monthly_expenses_avg:,.2f}", "Average monthly spending over the last 12 months"),
            ])
            render_metric_grid([
                ("Avg Monthly Profit", f"${snapshot.monthly_profit_avg:,.2f}", "Income minus expenses on average"),
                ("Current Retirement Balance", f"${snapshot.current_retirement_balance:,.2f}", "Starting balance used in the projection"),
            ])
            render_metric_grid([
                ("Current Asset Total", f"${snapshot.current_asset_total:,.2f}", "All assets combined from your file"),
            ])

    if not financial_df.empty:
        cashflow_chart = build_cashflow_chart(financial_df)
        if not cashflow_chart.empty:
            st.markdown("**Income, Spending, and Profit Trend**")
            st.line_chart(cashflow_chart.set_index("Month"))


def render_guidance_summary(guidance_text: str) -> None:
    clean_text = guidance_text.strip() if guidance_text else ""
    if clean_text:
        st.markdown(
            """
            <div style="padding: 1rem 1.1rem; border-radius: 1rem; border: 1px solid rgba(37,99,235,0.18); background: linear-gradient(135deg, rgba(239,246,255,0.96), rgba(255,255,255,0.98)); box-shadow: 0 10px 24px rgba(37,99,235,0.06);">
            """,
            unsafe_allow_html=True,
        )
        st.markdown(clean_text)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No guidance generated yet.")


def render_summary_technical(projection, financial_df: pd.DataFrame) -> None:
    projection_data = asdict(projection)
    with st.expander("Technical details", expanded=False):
        st.write("Projection math, inflation data, and the full data payload stay here for transparency without crowding the page.")
        with st.expander("See how the projection was calculated"):
            st.json(projection_data)
        with st.expander("See the inflation snapshot"):
            st.json(st.session_state.inflation_context)
        with st.expander("See the data dictionary"):
            render_data_dictionary()
        if not financial_df.empty:
            with st.expander("See the uploaded data preview"):
                st.dataframe(financial_df, use_container_width=True)


def generate_report_from_state():
    financial_df = st.session_state.financial_df
    inputs = st.session_state.inputs

    financial_snapshot = summarize_financial_csv(financial_df) if not financial_df.empty else None
    starting_balance = financial_snapshot.current_retirement_balance if financial_snapshot else 0.0

    projection = calculate_compound_growth(
        current_age=int(inputs["current_age"]),
        target_age=int(inputs["target_age"]),
        annual_contribution=float(inputs["annual_contribution"]),
        desired_income=float(inputs["desired_income"]),
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
    return projection


def main() -> None:
    inject_styles()
    render_hero()

    if "projection" not in st.session_state:
        st.session_state.projection = None
    if "guidance" not in st.session_state:
        st.session_state.guidance = ""
    if "inflation_context" not in st.session_state:
        st.session_state.inflation_context = {}
    if "page_view" not in st.session_state:
        st.session_state.page_view = "Enter Your Details"
    if "financial_df" not in st.session_state:
        st.session_state.financial_df = pd.DataFrame()
    if "inputs" not in st.session_state:
        st.session_state.inputs = {
            "current_age": 30,
            "target_age": 65,
            "annual_contribution": 10000.0,
            "desired_income": 70000.0,
        }
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    submitted = False

    with st.container():
        st.markdown('<div class="page-shell">', unsafe_allow_html=True)

        if st.session_state.page_view == "Enter Your Details":
            render_section_intro("Enter Your Details", "Start with a sample file or your own spreadsheet.")
            st.caption("Use the sample file if you want a quick template, or upload your own spreadsheet to get started.")
            template_csv = get_sample_template().to_csv(index=False)
            st.download_button(
                "Download sample spreadsheet",
                data=template_csv,
                file_name="sample_financial_data.csv",
                mime="text/csv",
                use_container_width=False,
            )

            with st.form("retirement_inputs"):
                left_panel, right_panel = st.columns([1, 1.2])
                with left_panel:
                    st.markdown("<div class='summary-panel'>", unsafe_allow_html=True)
                    st.markdown("<div class='panel-title'>What to enter</div>", unsafe_allow_html=True)
                    st.markdown("<div class='panel-subtitle'>Your age, your target age, how much you save each year, and the income you want in retirement.</div>", unsafe_allow_html=True)

                    default_inputs = st.session_state.inputs
                    left_top, right_top = st.columns(2)
                    with left_top:
                        current_age = st.number_input("Current Age", min_value=0, max_value=120, value=int(default_inputs["current_age"]), step=1)
                    with right_top:
                        target_age = st.number_input("Target Age", min_value=0, max_value=120, value=int(default_inputs["target_age"]), step=1)

                    left_bottom, right_bottom = st.columns(2)
                    with left_bottom:
                        annual_contribution = st.number_input("401(k) Contribution", min_value=0.0, value=float(default_inputs["annual_contribution"]), step=500.0)
                    with right_bottom:
                        desired_income = st.number_input("Desired Income", min_value=0.0, value=float(default_inputs["desired_income"]), step=1000.0)
                    st.markdown("</div>", unsafe_allow_html=True)

                with right_panel:
                    st.markdown("<div class='summary-panel'>", unsafe_allow_html=True)
                    st.markdown("<div class='panel-title'>Upload your spreadsheet</div>", unsafe_allow_html=True)
                    st.markdown("<div class='panel-subtitle'>Preview your data on the right so you can quickly check everything before moving on.</div>", unsafe_allow_html=True)
                    uploaded_file = st.file_uploader("Upload your spreadsheet", type=["csv", "xlsx", "xls"])
                    financial_df = load_financial_csv(uploaded_file)
                    st.session_state.financial_df = financial_df
                    if not financial_df.empty:
                        validation = validate_financial_csv(financial_df)
                        if validation.is_valid:
                            st.success("Your file loaded successfully.")
                        else:
                            st.warning("Your file loaded, but a few things need attention.")
                            for issue in validation.issues:
                                st.write(f"- {issue}")
                        st.dataframe(financial_df.head(5), use_container_width=True, height=180)
                        with st.expander("View full preview"):
                            st.dataframe(financial_df, use_container_width=True, height=320)
                        with st.expander("Data dictionary"):
                            render_data_dictionary()
                    else:
                        st.caption("Upload a spreadsheet to preview your financial data and unlock the summary.")
                    st.markdown("</div>", unsafe_allow_html=True)

                submit_left, submit_mid, submit_right = st.columns([1, 1, 1])
                with submit_mid:
                    submitted = st.form_submit_button("See My Summary")

        if submitted:
            st.session_state.inputs = {
                "current_age": int(current_age),
                "target_age": int(target_age),
                "annual_contribution": float(annual_contribution),
                "desired_income": float(desired_income),
            }
            st.session_state.projection = generate_report_from_state()
            st.session_state.page_view = "Summary"
            st.rerun()

        if st.session_state.page_view == "Summary":
            if st.session_state.projection is None:
                st.session_state.projection = generate_report_from_state()

            header_left, header_right = st.columns([6, 1])
            with header_left:
                render_section_intro("Summary", "Your retirement snapshot at a glance.")
            with header_right:
                st.write("")
                if st.button("← Back to Details", use_container_width=True):
                    st.session_state.page_view = "Enter Your Details"
                    st.rerun()

            projection = st.session_state.projection
            financial_df = st.session_state.financial_df
            if projection is None:
                st.info("Run the report from the inputs section to view projections and guidance.")
                return

            overview_tab, recommendations_tab, assistant_tab, technical_tab = st.tabs([
                "Overview",
                "Recommendations",
                "Planning Assistant",
                "Technical Details",
            ])

            with overview_tab:
                render_summary_overview(projection, financial_df, st.session_state.inflation_context)

            with recommendations_tab:
                st.subheader("Personalized Guidance")
                render_guidance_summary(st.session_state.guidance)

            with assistant_tab:
                assistant_header_left, assistant_header_right = st.columns([6, 1])
                with assistant_header_left:
                    st.subheader("Dynamic Planning Assistant")
                    st.caption("Ask follow-up questions to explore different retirement scenarios in real time.")
                with assistant_header_right:
                    st.write("")
                    if st.button("Clear", use_container_width=True):
                        st.session_state.chat_history = []
                        st.rerun()

                for message in st.session_state.chat_history:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])

                if chat_prompt := st.chat_input("Ask a what-if question (e.g., 'What if I increase my 401k contribution by $300/mo?')..."):
                    st.session_state.chat_history.append({"role": "user", "content": chat_prompt})
                    updated_inputs, updated_projection, answer = run_what_if_analysis(
                        user_text=chat_prompt,
                        current_inputs=st.session_state.inputs,
                        current_projection=st.session_state.projection,
                        financial_df=st.session_state.financial_df,
                        inflation_context=st.session_state.inflation_context,
                    )
                    st.session_state.inputs = updated_inputs
                    st.session_state.projection = updated_projection
                    render_scenario_summary(updated_inputs, updated_projection)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.rerun()

            with technical_tab:
                render_summary_technical(projection, financial_df)

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
