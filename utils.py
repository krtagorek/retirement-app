from __future__ import annotations

import os
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

    return RetirementProjection(
        current_age=current_age,
        target_age=target_age,
        annual_contribution=annual_contribution,
        desired_income=desired_income,
        starting_balance=starting_balance,
        years=years,
        annual_return_rate=annual_return_rate,
        projected_balance=balance,
        gap_to_target_income=max(desired_income - balance / max(years, 1), 0.0),
    )


def load_financial_csv(uploaded_file: Any) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    return pd.read_csv(uploaded_file)


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
                    "Do not claim to be a financial advisor."
                ),
            },
            {
                "role": "user",
                "content": f"Create concise retirement readiness guidance from this JSON:\n{prompt}",
            },
        ],
    )

    return response.output_text
