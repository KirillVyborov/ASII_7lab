import pytest
import pandas as pd
import pandera as pa
from scipy.stats import ks_2samp

from data_validation import load_titanic, validate_data, validate_with_great_expectations

# Числовые признаки, по которым проверяется дрейф
DRIFT_COLUMNS = ["Pclass", "Age", "SibSp", "Parch", "Fare"]


def _split_reference_current(df: pd.DataFrame):
    reference = df.sample(300, random_state=42)
    current = df.drop(reference.index)
    return reference, current


# --- 1. ТЕСТЫ ВАЛИДАЦИИ СХЕМЫ (PANDERA) ---

def test_valid_data_pandera():
    raw_data = load_titanic()
    validated_df = validate_data(raw_data)
    assert not validated_df.empty


def test_invalid_data_pandera():
    invalid_df = pd.DataFrame({
        "PassengerId": [1],
        "Survived": [0],
        "Pclass": [3],
        "Name": ["Test, Mr. Invalid"],
        "Sex": ["male"],
        "Age": [-5.0],  # Отрицательный возраст!
        "SibSp": [0],
        "Parch": [0],
        "Ticket": ["A/0"],
        "Fare": [7.25],
        "Cabin": [None],
        "Embarked": ["S"],
    })
    with pytest.raises(pa.errors.SchemaError):
        validate_data(invalid_df)


def test_invalid_category_pandera():
    df = load_titanic().head(5).copy()
    df.loc[0, "Pclass"] = 4  # Такого класса не существует
    with pytest.raises(pa.errors.SchemaError):
        validate_data(df)


# --- 2. ТЕСТЫ GREAT EXPECTATIONS ---

def test_great_expectations_checks():
    raw_data = load_titanic()
    assert validate_with_great_expectations(raw_data) is True


# --- 3. ТЕСТ НА ОБНАРУЖЕНИЕ ДРЕЙФА ДАННЫХ (KS-ТЕСТ) ---

def test_data_drift_detection():
    df = load_titanic()
    reference, current_normal = _split_reference_current(df)

    for col in DRIFT_COLUMNS:
        p_value = ks_2samp(reference[col].dropna(), current_normal[col].dropna()).pvalue
        assert p_value > 0.01, f"Обнаружен дрейф в признаке {col}"


def test_simulated_drift_failure():
    df = load_titanic()
    reference, current = _split_reference_current(df)

    # Имитируем дрейф признака: стоимость билета выросла в 3 раза
    current_drifted = current.copy()
    current_drifted["Fare"] *= 3.0

    p_value = ks_2samp(reference["Fare"], current_drifted["Fare"]).pvalue
    assert p_value < 0.05  # Дрейф успешно обнаружен
