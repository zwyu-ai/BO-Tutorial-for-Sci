import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hebo.optimizers.hebo import HEBO
from hebo.design_space.design_space import DesignSpace
from typing import Callable


# load data and construct oracle
def construct_oracle(
    data_path: str,
    impl: str = "random_forest"
) -> Callable[[pd.DataFrame], np.ndarray]:
    
    data = pd.read_csv(data_path)
    target = data["Target"]
    features = data.drop(columns=["Target"])

    # learn a decision tree model as oracle
    if impl == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100)
        model.fit(features, target)

        return lambda x: model.predict(x)
    else:
        raise ValueError(f"Unsupported implementation: {impl}")



def plot_results(log: list, color, label: str):
    targets = np.asarray(log)
    batch_means = targets.mean(axis=1)
    best_so_far = [targets[:i + 1].max() for i in range(targets.shape[0])]
    plt.plot(best_so_far, color=color,
             linestyle='-', marker='*', markersize=6, linewidth=2,
             label=f"{label} (best so far)")
    plt.plot(batch_means, color=color,
             linestyle=':', marker='o',
             label=f"{label} (batch mean)")
    plt.xlabel("Iteration")
    plt.ylabel("Target Value")


if __name__ == "__main__":
    # construct oracle model
    print("Constructing oracle model...")
    oracle = construct_oracle(data_path="HER_virtual_data.csv")
    print("Oracle model constructed.\n")

    # configure the parameter space
    param_names = [
        "AcidRed871_0gL", "L-Cysteine-50gL", "MethyleneB_250mgL", "NaCl-3M",
        "NaOH-1M", "P10-MIX1", "PVP-1wt", "RhodamineB1_0gL", "SDS-1wt",
        "Sodiumsilicate-1wt"
    ]
    param_space = DesignSpace().parse([
        dict(name=name, type='num', lb=0.0, ub=5.0) for name in param_names
    ])

    # configure hyper parameters
    n_iter = 100
    n_suggestions = 10

    # use the HEBO
    hebo = HEBO(space=param_space)
    log_hebo: list = []
    for i_iter in range(n_iter):
        suggestions = hebo.suggest(n_suggestions)
        targets = oracle(suggestions)
        hebo.observe(suggestions, -targets)  # HEBO minimizes the objective, so we negate the targets

        print(f"HEBO Iteration {i_iter + 1}/{n_iter}, mean target: {targets.mean():.4f}")
        log_hebo.append(targets.tolist())

    print("\n" + "=" * 50 + "\n")

    # baseline: random search
    log_random: list = []
    for i_iter in range(n_iter):
        suggestions = param_space.sample(n_suggestions)
        targets = oracle(suggestions)
        print(f"Random Iteration {i_iter + 1}/{n_iter}, mean target: {targets.mean():.4f}")
        log_random.append(targets.tolist())

    # plot results
    plt.figure(figsize=(8, 5))
    plot_results(log_hebo, color='b', label='HEBO')
    plot_results(log_random, color='r', label='Random Search')
    plt.legend()
    plt.title("Optimization of Photocatalytic Water Splitting")
    plt.grid()
    plt.tight_layout()
    plt.show()
