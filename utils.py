import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from hebo.optimizers.bo import BO
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")


def plot_regret(
        methods_data,
        title="Optimization Convergence",
        xlabel="Iteration",
        ylabel="Best Value Found",
        show_points=False,
        point_alpha=0.15,
        fig_path: str | None = None
):
    """
    Plots the mean cumulative minimum and standard error, with optional scatter
    of all points tried across all seeds.
    """
    plt.figure(figsize=(12, 7))

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#454645']

    for (label, data), color in zip(methods_data.items(), colors):
        num_seeds, num_iters = data.shape
        iters = np.arange(num_iters)

        # 1. Calculate cumulative minimum per seed
        cum_min = np.minimum.accumulate(data, axis=1)

        # 2. Calculate stats for the cumulative lines
        mean_cum = np.mean(cum_min, axis=0)
        std_err = np.std(cum_min, axis=0) / np.sqrt(num_seeds)

        # 3. Optional: Plot all points tried across all seeds
        if show_points:
            # We plot the data.T (transposed) so each column is a seed's points
            # This is faster than a loop and keeps iterations aligned on X
            plt.plot(iters, data.T, '.', color=color, alpha=point_alpha, markersize=4, markeredgewidth=0)
            # Add a dummy plot for the legend to represent "Points"
            plt.plot([], [], '.', color=color, alpha=0.6, label=f"{label} (Iter)")

        # 4. Plot mean cumulative line
        plt.plot(iters, mean_cum, label=f"{label} (Best)", lw=3, color=color)

        # 5. Plot shaded error region (Mean +/- 1 Standard Error)
        plt.fill_between(iters, mean_cum - std_err, mean_cum + std_err, color=color, alpha=0.2)

    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(xlabel, fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')

    plt.grid(True, alpha=0.3, linestyle='--')

    # Adjust legend to handle the point labels nicely
    plt.legend(loc='upper right', frameon=True, shadow=True, ncol=2 if show_points else 1)

    plt.tight_layout()

    if fig_path is not None:
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {fig_path}")

    plt.show()


def plot_combined_optimization_results(
        results_list,
        titles,
        ylabels,
        show_points=False,
        point_alpha=0.1,
        fig_path=None
):
    """
    Plots the mean cumulative minimum and standard error for multiple datasets
    side-by-side in a single row.
    """
    # Create a figure with 1 row and 3 columns
    fig, axes = plt.subplots(1, len(results_list), figsize=(6 * len(results_list), 6), sharex=False)
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#454645']

    if len(results_list) == 1:
        axes = [axes]  # Make it iterable

    for i, (methods_data, title, ylabel) in enumerate(zip(results_list, titles, ylabels)):
        ax = axes[i]

        for (label, data), color in zip(methods_data.items(), colors):
            num_seeds, num_iters = data.shape
            iters = np.arange(num_iters)

            # 1. Calculate cumulative minimum per seed (assuming minimization)
            # If your target is to maximize (e.g. yield), use np.maximum.accumulate
            cum_min = np.minimum.accumulate(data, axis=1)

            # 2. Calculate stats
            mean_cum = np.mean(cum_min, axis=0)
            std_err = np.std(cum_min, axis=0) / np.sqrt(num_seeds)

            # 3. Optional: Plot all points tried across all seeds
            if show_points:
                ax.plot(iters, data.T, '.', color=color, alpha=point_alpha, markersize=3, markeredgewidth=0)
                # Add a dummy plot for the legend to represent iterations if desired
                # ax.plot([], [], '.', color=color, alpha=0.6, label=f"{label} (Iter)")

            # 4. Plot mean cumulative line
            ax.plot(iters, mean_cum, label=f"{label}", lw=2.5, color=color)

            # 5. Plot shaded error region (Mean +/- 1 Standard Error)
            ax.fill_between(iters, mean_cum - std_err, mean_cum + std_err, color=color, alpha=0.2)

        # Formatting each subplot
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel("Iteration", fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add legend to each subplot or just the last one
        ax.legend(loc='upper right', frameon=True, shadow=True, fontsize='small')

    plt.tight_layout()

    if fig_path is not None:
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Combined plot saved to {fig_path}")

    plt.show()


def run_optimization(
        space: DesignSpace,
        oracle: Callable,
        num_iterations: int,
        random_seeds: list[int],
        random_samples: int = 10
):
    # ========== 1. HEBO Optimization ==========
    print("\n[1/3] Running HEBO Optimization...")
    hebo_config = {
        'lr': 5e-3,
        'num_epochs': 100,
        'verbose': False,
        'noise_lb': 1e-5,
        'optimizer': 'lbfgs',
        'pred_likeli': True,
        'warp': True,
    }
    hebo_history = []

    for seed in random_seeds:
        hebo = HEBO(space, model_name='gp', rand_sample=random_samples, model_config=hebo_config)
        hebo_trace = []

        for i in tqdm(range(num_iterations)):
            try:
                rec_x = hebo.suggest()
            except Exception as e:
                print(f"⚠️ HEBO suggest() failed at iteration {i} with error: {e}. Abort")
                break
            rec_y = oracle(rec_x)
            hebo_trace += rec_y.flatten().tolist()
            hebo.observe(rec_x, rec_y)
            best_y = hebo.y.min()

        hebo_history.append(hebo_trace)
        print(f"✓ HEBO seed {seed} completed. Final best: {hebo.y.min():.2f}")

    # ========== 2. Basic BO with LCB ==========
    print("\n[2/3] Running Basic BO (LCB)...")
    bo_history = []

    for seed in random_seeds:
        bo = BO(space, model_name='gp', rand_sample=random_samples)
        bo_trace = []

        for i in tqdm(range(num_iterations)):
            try:
                rec_x = bo.suggest()
            except Exception as e:
                print(f"⚠️ BO suggest() failed at iteration {i} with error: {e}. Abort")
                break
            rec_y = oracle(rec_x)
            bo_trace += rec_y.flatten().tolist()
            bo.observe(rec_x, rec_y)
            best_y = bo.y.min()

        bo_history.append(bo_trace)
        print(f"✓ BO seed {seed} completed. Final best: {bo.y.min():.2f}")

    # ========== 3. Random Search ==========
    print("\n[3/3] Running Random Search...")
    rs_history = []

    for seed in random_seeds:
        np.random.seed(seed)
        rs_trace = []
        for i in tqdm(range(num_iterations)):
            rec_x = space.sample(1)
            rec_y = oracle(rec_x)
            rs_trace += rec_y.flatten().tolist()
            best_y = min(rs_trace)

        rs_history.append(rs_trace)
        print(f"✓ Random Search seed {seed} completed. Final best: {min(rs_trace):.2f}")

    # ========== Results Summary ==========
    hebo_history = np.array(hebo_history)
    bo_history = np.array(bo_history)
    rs_history = np.array(rs_history)

    return hebo_history, bo_history, rs_history
