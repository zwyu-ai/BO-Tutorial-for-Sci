from typing import Callable

import numpy as np
import pandas as pd
from hebo.design_space.design_space import DesignSpace


def get_preprocessed_data(data_path: str = 'BH/BH_dataset.csv') -> pd.DataFrame:
    data = pd.read_csv(data_path)
    data["yield"] = data["yield"].max() - data["yield"]
    return data


def feature_selection(
    data_df: pd.DataFrame,
    importance_extractor: str = "random_forest",
    max_n: int = -1,
    max_cum_imp: float = 0,
    min_imp: float = 0,
) -> pd.DataFrame:
    """
    Extract imporant features based on feature importance.
    """
    target = data_df["yield"]
    features = data_df.drop(columns=["yield", "cost", "new_index"], errors='ignore')

    if importance_extractor == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100)
        model.fit(features, target)
        imp = model.feature_importances_
    elif importance_extractor == "decision_tree":
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor()
        model.fit(features, target)
        imp = model.feature_importances_
    elif importance_extractor == "lasso":
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=0.01)
        model.fit(features, target)
        imp = np.abs(model.coef_)
    else:
        raise ValueError(f"Unsupported importance extractor: {importance_extractor}")
    
    indices = np.argsort(imp)[::-1]

    # filter by max_n_feature
    if max_n > 0:
        indices = indices[:max_n]
    
    # filter by cumulative importance
    if max_cum_imp > 0:
        cum_imp = np.cumsum(imp[indices])
        indices = indices[cum_imp <= max_cum_imp]
    
    # filter by minimum importance
    if min_imp > 0:
        indices = indices[imp[indices] >= min_imp]


    print("Selected features:")
    for i, idx in enumerate(indices):
        print(f"{i+1}. {features.columns[idx]}: importance = {imp[idx]:.4f}")

    selected_features = features.columns[indices].tolist()
    selected_features.append("yield")
    
    return data_df[selected_features]


def get_design_space(data_df: pd.DataFrame) -> DesignSpace:
    params_columns = data_df.columns.tolist()
    for p in ['yield', 'cost', 'new_index']:
        if p in params_columns:
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


def prepare(
    data_path: str = 'examples/BH/BH_dataset.csv',
    oracle_impl: str = "random_forest",
    feature_selector: str | None = None,
    max_n: int = -1,
    max_cum_imp: float = 0,
    min_imp: float = 0,
    
):
    data = get_preprocessed_data(data_path)
    if feature_selector is not None:
        data = feature_selection(
            data,
            importance_extractor=feature_selector,
            max_n=max_n,
            max_cum_imp=max_cum_imp,
            min_imp=min_imp,
        )
    oracle = construct_oracle(data, impl=oracle_impl)
    design_space = get_design_space(data)
    return design_space, oracle
