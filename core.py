# core.py
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def read_data(dataset: str = "iris") -> pd.DataFrame:
    """
    Pretend to load some data (from disk, database, API, etc).
    Right now it just loads the iris dataset.
    """
    #time.sleep(0.0)  # simulate slow I/O

    if dataset == "iris":
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df["target"] = iris.target
        df["target_name"] = df["target"].map(
            {i: name for i, name in enumerate(iris.target_names)}
        )
        return df

    # fall back to a tiny random dataset
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "x": rng.normal(size=100),
            "y": rng.normal(size[100]),
        }
    )
    return df


def make_iris_plot(df: pd.DataFrame, x_col: str, y_col: str):
    """
    Build a plot object from the iris data.
    We'll return a simple Altair scatter plot.
    """
    import altair as alt  # imported here so core.py still imports without Altair installed

    time.sleep(0.1)  # pretend plot building is non-trivial

    chart = (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(
            x=x_col,
            y=y_col,
            color="target_name",
            tooltip=["target_name", x_col, y_col],
        )
        .properties(width=500, height=350)
    )
    return chart


@dataclass
class ModelResult:
    big: bool
    accuracy: float
    n_samples: int
    train_time_seconds: float


def run_model(df: pd.DataFrame, big: bool = False) -> ModelResult:
    """
    Fake 'expensive' model training.
    - big=False: quick run
    - big=True: pretend it's a big deep model
    """
    X = df.iloc[:, :4]  # first four numeric columns
    y = df["target"]

    # Simulate different runtimes
    pretend_seconds = 0.1 if not big else 5

    start = time.time()
    time.sleep(pretend_seconds)  # <-- this is the "expensive" part

    # Simple logistic regression as our 'model'
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    duration = time.time() - start

    return ModelResult(
        big=big,
        accuracy=float(acc),
        n_samples=len(df),
        train_time_seconds=duration,
    )
