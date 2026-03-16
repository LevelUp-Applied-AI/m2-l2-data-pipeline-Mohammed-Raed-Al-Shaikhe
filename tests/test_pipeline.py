"""
Lab 2 — Learner Test File

Write your own pytest tests here. You must implement at least 3 test functions:
  - test_load_data_returns_dataframe
  - test_clean_data_no_nulls
  - test_add_features_creates_revenue

The autograder will run your tests as part of the CI check.
"""

import pandas as pd
import numpy as np
import pytest
from pipeline import load_data, clean_data, add_features


# ─── Test 1 ───────────────────────────────────────────────────────────────────

def test_load_data_returns_dataframe():
    df = load_data("data/sales_records.csv")

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    expected_columns = [
        'date',
        'store_id',
        'product_category',
        'quantity',
        'unit_price',
        'payment_method'
    ]

    for col in expected_columns:
        assert col in df.columns


# ─── Test 2 ───────────────────────────────────────────────────────────────────

def test_clean_data_no_nulls():
    df = load_data("data/sales_records.csv")
    cleaned = clean_data(df)

    assert cleaned['quantity'].isna().sum() == 0
    assert cleaned['unit_price'].isna().sum() == 0


# ─── Test 3 ───────────────────────────────────────────────────────────────────

def test_add_features_creates_revenue():
    df = load_data("data/sales_records.csv")
    cleaned = clean_data(df)
    enriched = add_features(cleaned)

    assert 'revenue' in enriched.columns

    expected = enriched['quantity'] * enriched['unit_price']

    pd.testing.assert_series_equal(
        enriched['revenue'],
        expected,
        check_names=False
    )
