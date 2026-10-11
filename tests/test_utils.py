from __future__ import annotations

import unittest
from io import BytesIO

import pandas as pd

from utils import (
    calculate_compound_growth,
    load_financial_csv,
    parse_what_if_text,
    summarize_financial_csv,
    validate_financial_csv,
)


class TestRetirementUtils(unittest.TestCase):
    def test_calculate_compound_growth_uses_starting_balance(self) -> None:
        projection = calculate_compound_growth(
            current_age=45,
            target_age=50,
            annual_contribution=10000,
            desired_income=80000,
            annual_return_rate=0.07,
            starting_balance=100000,
        )

        self.assertEqual(projection.years, 5)
        self.assertAlmostEqual(projection.starting_balance, 100000)
        self.assertGreater(projection.projected_balance, 100000)
        self.assertAlmostEqual(projection.projected_retirement_income, projection.projected_balance * 0.04)
        self.assertAlmostEqual(projection.gap_to_target_income, max(80000 - projection.projected_retirement_income, 0.0))

    def test_validate_financial_csv_accepts_expected_columns(self) -> None:
        df = pd.DataFrame(
            {
                "month": ["2025-01-01", "2025-02-01"],
                "income": [9000, 9200],
                "expenses": [6500, 6400],
                "profit": [2500, 2800],
                "retirement_401k_balance": [75000, 76000],
                "roth_ira_balance": [18000, 18200],
                "taxable_investments": [42000, 43000],
                "cash_savings": [16000, 16500],
                "home_equity": [140000, 141000],
                "other_assets": [5000, 5000],
                "total_assets": [296000, 299700],
            }
        )

        validation = validate_financial_csv(df)

        self.assertTrue(validation.is_valid)
        self.assertEqual(validation.issues, [])

    def test_validate_financial_csv_allows_additional_columns(self) -> None:
        df = pd.DataFrame(
            {
                "month": ["2025-01-01"],
                "income": [9000],
                "expenses": [6500],
                "profit": [2500],
                "total_assets": [299700],
                "notes": ["extra column is okay"],
                "custom_metric": [123],
            }
        )

        validation = validate_financial_csv(df)

        self.assertTrue(validation.is_valid)
        self.assertEqual(validation.issues, [])

    def test_validate_financial_csv_flags_missing_required_columns(self) -> None:
        df = pd.DataFrame(
            {
                "month": ["2025-01-01"],
                "income": [9000],
                "profit": [2500],
                "total_assets": [299700],
            }
        )

        validation = validate_financial_csv(df)

        self.assertFalse(validation.is_valid)
        self.assertTrue(any("Missing required columns" in issue for issue in validation.issues))

    def test_load_financial_csv_returns_empty_frame_for_unsupported_file(self) -> None:
        unsupported = BytesIO(b"not a csv")
        unsupported.name = "notes.txt"

        loaded = load_financial_csv(unsupported)

        self.assertTrue(loaded.empty)

    def test_summarize_financial_csv_uses_latest_balance_and_totals(self) -> None:
        df = pd.DataFrame(
            {
                "month": ["2025-01-01", "2025-02-01"],
                "income": [9000, 9200],
                "expenses": [6500, 6400],
                "profit": [2500, 2800],
                "retirement_401k_balance": [75000, 76000],
                "roth_ira_balance": [18000, 18200],
                "taxable_investments": [42000, 43000],
                "cash_savings": [16000, 16500],
                "home_equity": [140000, 141000],
                "other_assets": [5000, 5000],
                "total_assets": [296000, 299700],
            }
        )

        snapshot = summarize_financial_csv(df)

        self.assertAlmostEqual(snapshot.monthly_income_avg, 9100.0)
        self.assertAlmostEqual(snapshot.monthly_expenses_avg, 6450.0)
        self.assertAlmostEqual(snapshot.monthly_profit_avg, 2650.0)
        self.assertAlmostEqual(snapshot.current_retirement_balance, 94200.0)
        self.assertAlmostEqual(snapshot.current_asset_total, 299700.0)

    def test_parse_what_if_text_handles_percent_change(self) -> None:
        current_inputs = {
            "current_age": 45,
            "target_age": 65,
            "annual_contribution": 12000.0,
            "desired_income": 70000.0,
        }
        projection = calculate_compound_growth(
            current_age=45,
            target_age=65,
            annual_contribution=12000.0,
            desired_income=70000.0,
            starting_balance=100000,
        )

        updated = parse_what_if_text("What if I increase my 401k contribution by 10%?", current_inputs, projection)
        self.assertAlmostEqual(updated["annual_contribution"], 13200.0)

    def test_parse_what_if_text_handles_stop_contributing(self) -> None:
        current_inputs = {
            "current_age": 45,
            "target_age": 65,
            "annual_contribution": 12000.0,
            "desired_income": 70000.0,
        }
        projection = calculate_compound_growth(
            current_age=45,
            target_age=65,
            annual_contribution=12000.0,
            desired_income=70000.0,
            starting_balance=100000,
        )

        updated = parse_what_if_text("What if I stop contributing to my 401k?", current_inputs, projection)
        self.assertEqual(updated["annual_contribution"], 0.0)


if __name__ == "__main__":
    unittest.main()
