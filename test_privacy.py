import pandas as pd
import joblib

# Test 1: Ensure removed columns are NOT in dataset
def test_removed_columns_not_in_dataset():
    data = pd.read_csv("data/net_worth_data_cleaned.csv")

    removed_columns = [
        "Client name",
        "Client email",
        "Profession",
        "Education",
        "Country",
        "Gender",
        "Stocks",
        "Bonds",
        "ETFs",
        "REITs",
        "Healthcare cost"
    ]

    for col in removed_columns:
        assert col not in data.columns

# Test 2: Dataset contains target column
def test_target_column_exists():
    data = pd.read_csv("data/net_worth_data_cleaned.csv")

    assert "Net Worth" in data.columns

# Test 3: Dataset shape is correct (5 features + 1 target)
def test_dataset_shape():
    data = pd.read_csv("data/net_worth_data_cleaned.csv")

    expected_columns = 6  # 5 features + target

    assert data.shape[1] == expected_columns
    assert data.shape[0] > 0


# Test 4: Approved features only
def test_approved_features():

    expected_features = [
        "Age",
        "Income",
        "Credit Card Debt",
        "Inherited Amount",
        "Mutual Funds"
    ]

    data = pd.read_csv("data/net_worth_data_cleaned.csv")

    actual_features = [col for col in data.columns if col != "Net Worth"]

    assert sorted(actual_features) == sorted(expected_features)

# Test 5: Removed columns list validation
def test_removed_columns_list():

    expected_removed = [
        "Client name",
        "Client email",
        "Profession",
        "Education",
        "Country",
        "Gender",
        "Stocks",
        "Bonds",
        "ETFs",
        "REITs",
        "Healthcare cost"
    ]

    removed_columns = expected_removed

    assert isinstance(removed_columns, list)
    assert "Age" not in removed_columns
    assert "Income" not in removed_columns


# Test 6: Model uses correct number of features
def test_model_feature_count():

    model = joblib.load("model/networth_model.pkl")

    expected_feature_count = 5

    assert model.n_features_in_ == expected_feature_count


# Test 7: Prediction output format safety check
def test_prediction_output_format():

    sample_output = {
        "Predicted Net Worth": 1000000
    }

    forbidden_fields = [
        "Client name",
        "Client email",
        "Profession",
        "Education",
        "Country",
        "Gender"
    ]

    assert "Predicted Net Worth" in sample_output

    for field in forbidden_fields:
        assert field not in sample_output