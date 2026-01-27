from typing import Callable

import numpy as np
import pandas as pd
from hebo.design_space.design_space import DesignSpace


def get_preprocessed_data(data_path: str = 'examples/HER/HER_virtual_data.csv') -> pd.DataFrame:
    data = pd.read_csv(data_path)
    data["Target"] = data["Target"].max() - data["Target"]
    return data


def get_design_space() -> DesignSpace:
    params = [
        {'name': 'AcidRed871_0gL', 'type': 'num', 'lb': 0, 'ub': 5},
        {'name': 'L-Cysteine-50gL', 'type': 'num', 'lb': 0, 'ub': 5},
        {'name': 'MethyleneB_250mgL', 'type': 'num', 'lb': 0, 'ub': 5},
        {'name': 'NaCl-3M', 'type': 'num', 'lb': 0, 'ub': 5},
        {'name': 'NaOH-1M', 'type': 'num', 'lb': 0, 'ub': 5},
        {'name': 'P10-MIX1', 'type': 'num', 'lb': 0, 'ub': 5},
        {'name': 'PVP-1wt', 'type': 'num', 'lb': 0, 'ub': 5},
        {'name': 'RhodamineB1_0gL', 'type': 'num', 'lb': 0, 'ub': 5},
        {'name': 'SDS-1wt', 'type': 'num', 'lb': 0, 'ub': 5},
        {'name': 'Sodiumsilicate-1wt', 'type': 'num', 'lb': 0, 'ub': 5},
    ]
    space = DesignSpace().parse(params)
    return space


def construct_oracle(data_df: pd.DataFrame, impl: str = "random_forest") -> Callable[[pd.DataFrame], np.ndarray]:
    target = data_df["Target"]
    features = data_df.drop(columns=["Target"])

    # learn a decision tree model as oracle
    if impl == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100)
        model.fit(features, target)

        return lambda x: model.predict(x).reshape(-1, 1)
    else:
        raise ValueError(f"Unsupported implementation: {impl}")


def prepare(
    data_path: str = 'examples/HER/HER_virtual_data.csv',
    oracle_impl: str = "random_forest"
):
    data = get_preprocessed_data(data_path)
    oracle = construct_oracle(data, impl=oracle_impl)
    design_space = get_design_space()
    return design_space, oracle
