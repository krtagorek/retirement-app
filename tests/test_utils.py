from __future__ import annotations

import unittest

import pandas as pd

from utils import calculate_compound_growth, summarize_financial_csv, validate_financial_csv


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


if __name__ == "__main__":
    unittest.main()
