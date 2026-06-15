import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from DataAnalysis import load_data, clean_data, analyze


def test_data_loads():
    data = load_data()
    assert data is not None


def test_clean_data_removes_nulls():
    data = load_data()
    cleaned = clean_data(data)
    assert cleaned.isnull().sum().sum() == 0


def test_columns_exist():
    data = load_data()
    cleaned = clean_data(data)

    assert "Net Worth" in cleaned.columns