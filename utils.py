from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from fredapi import Fred
from langgraph.graph import END, StateGraph
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


@dataclass
class RetirementProjection:
    current_age: int
    target_age: int
    annual_contribution: float
    desired_income: float
    starting_balance: float
    years: int
    annual_return_rate: float
    projected_balance: float
    projected_retirement_income: float
    gap_to_target_income: float


@dataclass
class CsvValidationResult:
    is_valid: bool
    issues: list[str]
    columns: list[str]


@dataclass
class FinancialSnapshot:
    monthly_income_avg: float
    monthly_expenses_avg: float
    monthly_profit_avg: float
    current_retirement_balance: float
    current_asset_total: float
    retirement_balance_columns: list[str]


def calculate_compound_growth(
    current_age: int,
    target_age: int,
    annual_contribution: float,
    desired_income: float,
    annual_return_rate: float = 0.07,
    starting_balance: float = 0.0,
) -> RetirementProjection:
    years = max(target_age - current_age, 0)
    balance = float(starting_balance)

    for _ in range(years):
        balance = balance * (1 + annual_return_rate) + annual_contribution

    projected_retirement_income = balance * 0.04
    return RetirementProjection(
        current_age=current_age,
        target_age=target_age,
        annual_contribution=annual_contribution,
        desired_income=desired_income,
        starting_balance=starting_balance,
        years=years,
        annual_return_rate=annual_return_rate,
        projected_balance=balance,
        projected_retirement_income=projected_retirement_income,
        gap_to_target_income=max(desired_income - projected_retirement_income, 0.0),
    )


def load_financial_csv(uploaded_file: Any) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    filename = getattr(uploaded_file, "name", "").lower()
    try:
        if filename.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        return pd.read_csv(uploaded_file)
    except Exception:
        return pd.DataFrame()


def validate_financial_csv(df: pd.DataFrame) -> CsvValidationResult:
    if df.empty:
        return CsvValidationResult(False, ["CSV is empty or missing."], [])

    issues: list[str] = []
    required_columns = {"month", "income", "expenses", "profit", "total_assets"}
    normalized_columns = {column.strip().lower() for column in df.columns}
    missing = sorted(required_columns - normalized_columns)
    if missing:
        issues.append(f"Missing required columns: {', '.join(missing)}")

    if not df.columns.is_unique:
        issues.append("CSV contains duplicate column names.")

    if df.isna().all(axis=1).any():
        issues.append("CSV contains fully empty rows.")

    numeric_columns = ["income", "expenses", "profit", "total_assets"]
    for column in numeric_columns:
        if column in normalized_columns:
            original = next(name for name in df.columns if name.strip().lower() == column)
            if not pd.api.types.is_numeric_dtype(df[original]):
                issues.append(f"Column '{original}' must be numeric.")

    return CsvValidationResult(is_valid=not issues, issues=issues, columns=list(df.columns))


def summarize_financial_csv(df: pd.DataFrame) -> FinancialSnapshot:
    normalized = {column.strip().lower(): column for column in df.columns}
    retirement_balance_columns = [
        normalized[column]
        for column in ["retirement_401k_balance", "roth_ira_balance"]
        if column in normalized
    ]
    asset_columns = [
        normalized[column]
        for column in ["retirement_401k_balance", "roth_ira_balance", "taxable_investments", "cash_savings", "home_equity", "other_assets"]
        if column in normalized
    ]

    return FinancialSnapshot(
        monthly_income_avg=float(df[normalized["income"]].mean()) if "income" in normalized else 0.0,
        monthly_expenses_avg=float(df[normalized["expenses"]].mean()) if "expenses" in normalized else 0.0,
        monthly_profit_avg=float(df[normalized["profit"]].mean()) if "profit" in normalized else 0.0,
        current_retirement_balance=float(df[retirement_balance_columns].iloc[-1].sum()) if retirement_balance_columns else 0.0,
        current_asset_total=float(df[normalized["total_assets"]].iloc[-1]) if "total_assets" in normalized else 0.0,
        retirement_balance_columns=retirement_balance_columns,
    )


def get_cpi_inflation_context() -> dict[str, Any]:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        return {
            "available": False,
            "message": "FRED_API_KEY is not configured.",
            "cpi_series": None,
        }

    fred = Fred(api_key=api_key)
    series = fred.get_series("CPIAUCSL", observation_start="2020-01-01")
    latest_value = float(series.iloc[-1]) if not series.empty else None
    previous_value = float(series.iloc[-13]) if len(series) >= 13 else None

    inflation_yoy = None
    if latest_value and previous_value:
        inflation_yoy = ((latest_value - previous_value) / previous_value) * 100

    return {
        "available": True,
        "series_id": "CPIAUCSL",
        "latest_value": latest_value,
        "previous_year_value": previous_value,
        "inflation_yoy_percent": inflation_yoy,
        "series_points": len(series),
    }


def run_langgraph_inflation_agent() -> dict[str, Any]:
    def fetch_inflation(_: dict[str, Any]) -> dict[str, Any]:
        return {"inflation_context": get_cpi_inflation_context()}

    graph = StateGraph(dict)
    graph.add_node("fetch_inflation", fetch_inflation)
    graph.set_entry_point("fetch_inflation")
    graph.add_edge("fetch_inflation", END)
    compiled = graph.compile()
    return compiled.invoke({})


def synthesize_guidance(
    projection: RetirementProjection,
    inflation_context: dict[str, Any],
    dataframe_summary: dict[str, Any] | None = None,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY is not configured. Guidance generation is unavailable."

    client = OpenAI(api_key=api_key)

    prompt = {
        "projection": {
            "current_age": projection.current_age,
            "target_age": projection.target_age,
            "annual_contribution": projection.annual_contribution,
            "desired_income": projection.desired_income,
            "starting_balance": projection.starting_balance,
            "years": projection.years,
            "annual_return_rate": projection.annual_return_rate,
            "projected_balance": round(projection.projected_balance, 2),
            "gap_to_target_income": round(projection.gap_to_target_income, 2),
        },
        "inflation_context": inflation_context,
        "dataframe_summary": dataframe_summary or {},
    }

    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        input=[
            {
                "role": "system",
                "content": (
                    "You are a fiduciary-style retirement planning assistant. "
                    "Give practical, cautious, non-personalized educational guidance. "
                    "Do not claim to be a financial advisor. "
                    "Return the answer in clean Markdown with the sections: Summary, What’s working, "
                    "What to consider, and Next steps. Use short bullet points under each section."
                ),
            },
            {
                "role": "user",
                "content": f"Create concise retirement readiness guidance from this JSON. Keep it under 250 words and format it as Markdown with clear headings and bullet points:\n{prompt}",
            },
        ],
    )

    return response.output_text


def initialize_chat_state() -> None:
    if "chat_history" not in globals():
        pass


def parse_what_if_text(
    user_text: str,
    current_inputs: dict[str, Any],
    current_projection: RetirementProjection,
) -> dict[str, Any]:
    updated = dict(current_inputs)

    text = user_text.lower()
    if any(phrase in text for phrase in ["stop contributing", "stop my 401k", "stop my 401(k)", "set contribution to zero", "zero out my 401k", "remove my 401k", "remove contribution"]):
        updated["annual_contribution"] = 0.0

    patterns = {
        "annual_contribution": [
            r"(?:add|increase|raise|boost|put in)\s+(?:an?\s+)?(?:extra\s+)?(?:\$?)([\d,]+(?:\.\d+)?)\s*(?:more\s+)?(?:/\s*mo|\s*per\s*month|\s*monthly|\s*mo|\s*a\s*month)?\s*(?:to|into|toward|in|for)?\s*(?:my\s+)?(?:401k(?:\s+contribution)?|401\(k\)(?:\s+contribution)?|retirement|contribution)?",
            r"(?:increase|raise|add|change|lower|decrease|reduce|set)\s+(?:my\s+)?(?:401k(?:\s+contribution)?|401\(k\)(?:\s+contribution)?|contribution)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*%",
            r"(?:increase|raise|add|change|lower|decrease|reduce|set)\s+(?:my\s+)?(?:401k(?:\s+contribution)?|401\(k\)(?:\s+contribution)?|contribution)\s+(?:by\s+)?\$?([\d,]+(?:\.\d+)?)\s*(?:/\s*mo|\s*per\s*month|\s*monthly|\s*mo)?",
            r"\$([\d,]+(?:\.\d+)?)\s*(?:/\s*mo|\s*per\s*month|\s*monthly|\s*mo)\s*(?:more|less)?\s*(?:to|toward)?\s*(?:my\s+)?(?:401k|401\(k\)|contribution)",
        ],
        "target_age": [
            r"(?:retire|retirement|target age)\s+(?:at|to|by)\s*(\d{2})",
            r"(?:work|delay)\s+(?:until|to)\s*(\d{2})",
            r"(?:retire|retirement)\s+(?:by|in)\s+(\d+)\s+years?",
        ],
    }

    for field, field_patterns in patterns.items():
        for pattern in field_patterns:
            match = re.search(pattern, text)
            if match:
                value = float(match.group(1).replace(",", ""))
                if field == "annual_contribution":
                    if "%" in pattern:
                        if any(word in text for word in ["decrease", "reduce", "lower"]):
                            value = max(current_projection.annual_contribution * (1 - value / 100.0), 0.0)
                        else:
                            value = current_projection.annual_contribution * (1 + value / 100.0)
                        updated[field] = value
                        break
                    is_monthly_amount = bool(
                        re.search(r"(?:/\s*mo|\s*per\s*month|\s*monthly|\s*mo|\s*a\s*month)", pattern)
                        or re.search(r"(?:/\s*mo|\s*per\s*month|\s*monthly|\s*mo|\s*a\s*month)", text)
                    )
                    if is_monthly_amount:
                        value *= 12
                    if any(word in text for word in ["decrease", "reduce", "lower", "less"]):
                        value = max(current_projection.annual_contribution - value, 0.0)
                    elif any(word in text for word in ["increase", "raise", "boost", "add", "more"]):
                        value = current_projection.annual_contribution + value
                    elif re.search(r"(?:add|increase|raise|boost|put in)\s+.*?(?:extra\s+)?(?:\$?)([\d,]+(?:\.\d+)?)\s*(?:/\s*mo|\s*per\s*month|\s*monthly|\s*mo|\s*a\s*month)?", text):
                        pass
                    updated[field] = value
                elif field == "target_age":
                    if "year" in pattern:
                        updated[field] = min(max(current_projection.current_age + int(value), current_projection.current_age), 120)
                    else:
                        updated[field] = int(value)
                break

    return updated


def run_what_if_analysis(
    user_text: str,
    current_inputs: dict[str, Any],
    current_projection: RetirementProjection,
    financial_df: pd.DataFrame,
    inflation_context: dict[str, Any],
) -> tuple[dict[str, Any], RetirementProjection, str]:
    updated_inputs = parse_what_if_text(user_text, current_inputs, current_projection)
    financial_snapshot = summarize_financial_csv(financial_df) if not financial_df.empty else None
    starting_balance = financial_snapshot.current_retirement_balance if financial_snapshot else 0.0

    updated_projection = calculate_compound_growth(
        current_age=int(updated_inputs.get("current_age", current_projection.current_age)),
        target_age=int(updated_inputs.get("target_age", current_projection.target_age)),
        annual_contribution=float(updated_inputs.get("annual_contribution", current_projection.annual_contribution)),
        desired_income=float(updated_inputs.get("desired_income", current_projection.desired_income)),
        starting_balance=starting_balance,
    )

    contribution_delta = float(updated_projection.annual_contribution) - float(current_projection.annual_contribution)
    if abs(contribution_delta) >= 1:
        delta_description = f"an extra ${abs(contribution_delta):,.2f} per year"
    else:
        delta_description = "the current contribution level"

    contribution_change_summary = (
        f"Contribution updated from ${current_projection.annual_contribution:,.2f}/year "
        f"to ${updated_projection.annual_contribution:,.2f}/year."
    )

    factual_summary = (
        f"Updated scenario: age {updated_projection.current_age} to {updated_projection.target_age}, "
        f"starting balance ${updated_projection.starting_balance:,.2f}, annual contribution ${updated_projection.annual_contribution:,.2f}, "
        f"projected balance ${updated_projection.projected_balance:,.2f}, gap ${updated_projection.gap_to_target_income:,.2f}."
    )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
    if client is None:
        answer = (
            f"{factual_summary}\n{contribution_change_summary}\n\n"
            f"That means {delta_description} changes the long-term outlook, and you still appear to have a gap of "
            f"${updated_projection.gap_to_target_income:,.2f}."
        )
        return updated_inputs, updated_projection, answer

    payload = {
        "user_question": user_text,
        "updated_inputs": updated_inputs,
        "projection": {
            "current_age": updated_projection.current_age,
            "target_age": updated_projection.target_age,
            "annual_contribution": updated_projection.annual_contribution,
            "desired_income": updated_projection.desired_income,
            "starting_balance": updated_projection.starting_balance,
            "years": updated_projection.years,
            "projected_balance": round(updated_projection.projected_balance, 2),
            "gap_to_target_income": round(updated_projection.gap_to_target_income, 2),
        },
        "inflation_context": inflation_context,
    }

    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        input=[
            {
                "role": "system",
                "content": (
                    "You are a concise retirement planning assistant. "
                    "Explain the updated result in plain English, mention the changed input(s), "
                    "and keep the answer brief and friendly. Do not change any numeric values. "
                    "Start with the exact factual summary provided by the app."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"factual_summary": factual_summary, **payload}),
            },
        ],
    )

    answer = f"{factual_summary}\n{contribution_change_summary}\n\n{response.output_text.strip()}"
    return updated_inputs, updated_projection, answer
