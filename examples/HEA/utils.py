import numpy as np
from numpy.typing import NDArray
import pandas as pd
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from typing import Callable

EPS = 1e-12

type _Vec = NDArray[np.floating]

l: _Vec = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
h: _Vec = np.array([0.35, 0.35, 0.35, 0.35, 0.35])


def _tail_sums(l: _Vec, h: _Vec):
    # l,h are arrays length n
    n = len(l)
    L = np.zeros(n + 2)
    H = np.zeros(n + 2)
    # fill 1-indexed tails: L[k] = sum_{i=k}^n l_i
    for k in range(n, 0, -1):
        L[k] = L[k+1] + l[k-1]
        H[k] = H[k+1] + h[k-1]
    return L, H

def _phi_inv(y: _Vec, l: _Vec, h: _Vec):
    # y length n-1 -> returns x length n
    n = len(l)
    assert len(y) == n-1
    L, H = _tail_sums(l, h)
    x = np.zeros(n)
    T = 1.0
    for k in range(1, n):
        a = max(l[k-1], T - H[k+1])
        b = min(h[k-1], T - L[k+1])
        if b + EPS < a:
            raise ValueError(f"Infeasible at index {k}: empty interval [{a},{b}]")
        # if degenerate interval b==a then x_k = a regardless of y_k
        xk = a + (0.0 if b - a <= EPS else y[k-1] * (b - a))
        x[k-1] = xk
        T -= xk
    x[-1] = T
    # final feasibility check (with tolerance)
    if not (l[-1] - EPS <= x[-1] <= h[-1] + EPS):
        raise ValueError("Final component out of bounds; numerical issue or infeasible y")
    return x

def _phi(x: np.ndarray, l: _Vec, h: _Vec):
    # x length n -> returns y length n-1
    n = len(l)
    assert len(x) == n
    L, H = _tail_sums(l, h)
    y = np.zeros(n-1)
    T = 1.0
    for k in range(1, n):
        a = max(l[k-1], T - H[k+1])
        b = min(h[k-1], T - L[k+1])
        xk = x[k-1]
        if not (a - EPS <= xk <= b + EPS):
            raise ValueError(f"x not in feasible interval at index {k}: {xk} not in [{a},{b}]")
        if b - a <= EPS:
            y[k-1] = 0.0
        else:
            y[k-1] = (xk - a) / (b - a)
        T -= xk
    # final check x_n == T and in bounds
    if not (abs(x[-1] - T) <= 1e-9 and l[-1] - EPS <= x[-1] <= h[-1] + EPS):
        raise ValueError("Final component check failed")
    return y


def construct_design_space():
    space = DesignSpace().parse([{'name' : 'x1', 'type' : 'num', 'lb' : 0, 'ub' : 1}, 
                                {'name' : 'x2', 'type' : 'num', 'lb' : 0, 'ub' : 1},
                                {'name' : 'x3', 'type' : 'num', 'lb' : 0, 'ub' : 1},
                                {'name' : 'x4', 'type' : 'num', 'lb' : 0, 'ub' : 1}])
    return space


def get_processed_data():
    data = pd.read_excel('./data/oracle_data.xlsx')
    data = data[['Co', 'Fe', 'Mn', 'V', 'Cu', 'target']]
    return data


def design_to_raw(data: pd.DataFrame) -> pd.DataFrame:
    """
    Converts design space representation to raw variable representation.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing design variables (x1, x2, x3, x4)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing raw variables (Co, Fe, Mn, V, Cu)
    """

    raw_data = []
    
    # check if the columns are in the correct order
    if list(data.columns) != ['x1', 'x2', 'x3', 'x4']:
        raise ValueError("Input DataFrame must have columns ['x1', 'x2', 'x3', 'x4'] in that order.")

    for _, row in data.iterrows():
        x = np.asarray(row.values)
        y = _phi_inv(x, l, h)
        raw_data.append(y)
    raw_df = pd.DataFrame(raw_data, columns=['Co', 'Fe', 'Mn', 'V', 'Cu'])
    return raw_df


def raw_to_design(data: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw variable representation to design space representation.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing raw variables (Co, Fe, Mn, V, Cu)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing design variables (x1, x2, x3, x4)
    """

    design_data = []
    
    # check if the columns are in the correct order
    if list(data.columns) != ['Co', 'Fe', 'Mn', 'V', 'Cu']:
        raise ValueError("Input DataFrame must have columns ['Co', 'Fe', 'Mn', 'V', 'Cu'] in that order.")

    for _, row in data.iterrows():
        y = np.asarray(row.values)
        x = _phi(y, l, h)
        design_data.append(x)
    design_df = pd.DataFrame(design_data, columns=['x1', 'x2', 'x3', 'x4'])
    return design_df


def _wrap_raw_oracle(oracle: Callable[[pd.DataFrame], np.ndarray],
                     max_target: float) -> Callable[[pd.DataFrame], np.ndarray]:
    """
    Wraps a raw variable oracle to accept design space inputs.

    Parameters:
    -----------
    oracle : Callable[[pd.DataFrame], np.ndarray]
        Oracle function that takes raw variable DataFrame and returns predictions.
    
    Returns:
    --------
    Callable[[pd.DataFrame], np.ndarray]
        Oracle function that takes design variable DataFrame and returns regret values.
    """

    def design_oracle(design_df: pd.DataFrame) -> np.ndarray:
        raw_df = design_to_raw(design_df)
        preds = oracle(raw_df).reshape(-1, 1)
        return max_target - preds

    return design_oracle


def construct_oracle(data_path: str, impl: str = "random_forest") -> Callable[[pd.DataFrame], np.ndarray]:
    data = pd.read_excel(data_path)
    data = data[['Co', 'Fe', 'Mn', 'V', 'Cu', 'target']]
    max_target = float(data['target'].max())

    if impl == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100)
        features = data.drop(columns=["target"])
        target = data["target"]
        model.fit(features, target)
        return _wrap_raw_oracle(model.predict, max_target)
    elif impl == "autogluon":
        from .oracle import Oracle
        oracle_model = Oracle(model_dir='datasets/HEA/AutogluonModels/ag-20251209_102532')
        return _wrap_raw_oracle(oracle_model, max_target)
    else:
        raise ValueError(f"Unknown implementation: {impl}")


def create_problem(
    data_path: str = "example/HEA/data/oracle_data.xlsx",
    oracle_impl: str = "random_forest"
):
    oracle = construct_oracle(data_path, impl=oracle_impl)
    design_space = construct_design_space()
    return design_space, oracle
