from typing import Callable

import numpy as np
import pandas as pd
from hebo.design_space.design_space import DesignSpace


def get_preprocessed_data(data_path: str = 'BH/BH_dataset.csv') -> pd.DataFrame:
    data = pd.read_csv(data_path)
    data["yield"] = data["yield"].max() - data["yield"]
    return data


def get_design_space(data_df: pd.DataFrame) -> DesignSpace:
    params_columns = data_df.columns.tolist()
    for p in ['yield', 'cost', 'new_index']:
        params_columns.remove(p)

    space = DesignSpace().parse([
        {'name': p, 'type': 'num', 'lb': data_df[p].min(), 'ub': data_df[p].max()} for p in params_columns
    ])
    return space


def construct_oracle(
        data_df: pd.DataFrame,
        impl: str = "random_forest"
) -> Callable[[pd.DataFrame], np.ndarray]:
    data_df.drop(columns=["cost", "new_index"], inplace=True, errors='ignore')
    target = data_df["yield"]
    features = data_df.drop(columns=["yield"])

    # learn a decision tree model as oracle
    if impl == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100)
        model.fit(features, target)

        return lambda x: model.predict(x).reshape(-1, 1)
    else:
        raise ValueError(f"Unsupported implementation: {impl}")
