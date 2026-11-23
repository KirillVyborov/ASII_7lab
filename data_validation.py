from pathlib import Path

import great_expectations as gx
import pandas as pd
import pandera as pa
from pandera import Column, Check

DATA_PATH = Path(__file__).parent / "titanic.csv"

# Декларативная схема валидации Pandera для датасета Titanic
titanic_schema = pa.DataFrameSchema(
    {
        "PassengerId": Column(int, Check.ge(1), unique=True),
        "Survived": Column(int, Check.isin([0, 1])),
        "Pclass": Column(int, Check.isin([1, 2, 3])),
        "Name": Column(str),
        "Sex": Column(str, Check.isin(["male", "female"])),
        "Age": Column(float, Check.in_range(0.0, 100.0), nullable=True),
        "SibSp": Column(int, Check.in_range(0, 8)),
        "Parch": Column(int, Check.in_range(0, 9)),
        "Ticket": Column(str),
        "Fare": Column(float, Check.ge(0.0)),
        "Cabin": Column(str, nullable=True),
        "Embarked": Column(str, Check.isin(["S", "C", "Q"]), nullable=True),
    },
    strict=True,
)


def load_titanic(path: Path = DATA_PATH) -> pd.DataFrame:
    """Загружает датасет Titanic из CSV."""
    return pd.read_csv(path)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Проверяет DataFrame по схеме и выбрасывает SchemaError при нарушениях."""
    return titanic_schema.validate(df)


def validate_with_great_expectations(df: pd.DataFrame) -> bool:
    """Проверяет качественные правила (Expectations) с помощью Great Expectations.

    Используется API Great Expectations 1.x (ge.from_pandas из старых версий
    в 1.x удалён).
    """
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("titanic_source")
    asset = data_source.add_dataframe_asset("titanic_asset")
    batch_definition = asset.add_batch_definition_whole_dataframe("titanic_batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    # 1. Проверка существования колонок
    res1 = batch.validate(gx.expectations.ExpectColumnToExist(column="Sex"))
    # 2. Проверка отсутствия пропусков: в Fare пропусков нет
    res2 = batch.validate(gx.expectations.ExpectColumnValuesToNotBeNull(column="Fare"))
    # 3. Пропуски в Age допустимы не более чем в 25% строк (в датасете ~20%)
    res3 = batch.validate(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="Age", mostly=0.75)
    )
    # 4. Значения Age находятся в физически допустимом интервале
    res4 = batch.validate(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="Age", min_value=0.0, max_value=100.0
        )
    )

    return all(res.success for res in (res1, res2, res3, res4))
