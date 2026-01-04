import numpy as np
import pandas as pd
import warnings
import dataclasses as dc
import os

from functools import partial
from utils import run_optimization, plot_combined_optimization_results
from typing import Callable
from hebo.design_space.design_space import DesignSpace

from OER.utils import prepare as prepare_OER
from HER.utils import prepare as prepare_HER
from BH.utils import prepare as prepare_BH
from HEA.utils import prepare as prepare_HEA

type OracleFn = Callable[[pd.DataFrame], np.ndarray]
type PrepareFn = Callable[..., tuple[DesignSpace, OracleFn]]


warnings.filterwarnings("ignore")


@dc.dataclass
class Config:
    title: str
    prepare_fn: PrepareFn
    num_iterations: int = 100
    num_seeds: int = 1
    random_samples: int = 10
    y_label: str = "y"
    result_path: str | None = None
    
    def run(self, ignore_if_exists: bool = False):
        results: dict[str, np.ndarray]
        
        if ignore_if_exists and self.result_path is not None and os.path.exists(self.result_path):
            print(f"✓ Result file {self.result_path} exists, skipping...")
            # load and return
            loaded = np.load(self.result_path, allow_pickle=True)
            # ensure that loaded is a dict and has desired keys
            results = {
                'HEBO': loaded['HEBO'],
                'BO (LCB)': loaded['BO (LCB)'],
                'Random Search': loaded['Random Search']
            }
        
        else:
            print("=" * 70)
            print(self.title)
            print("=" * 70)

            design_space, oracle = self.prepare_fn()
            hebo_history, bo_history, rs_history = run_optimization(
                space=design_space,
                oracle=oracle,
                num_iterations=self.num_iterations,
                random_seeds=list(range(0, self.num_seeds)),
                random_samples=self.random_samples
            )
            results = {
                'HEBO': hebo_history,
                'BO (LCB)': bo_history,
                'Random Search': rs_history
            }

            if self.result_path is not None:
                # make parent
                os.makedirs(os.path.dirname(self.result_path), exist_ok=True)
                np.savez(self.result_path, **results)

        return results



if __name__ == '__main__':
    configs = [
        Config(
            title="HER Optimization",
            prepare_fn=prepare_HER,
            num_iterations=200,
            num_seeds=16,
            random_samples=20,
            result_path="results/HER_bo_results.npz",
            y_label=r"Regret ($\mathrm{yield}^* - \mathrm{yield}$)",
        ),
        Config(
            title="HEA Nanozyme Optimization",
            prepare_fn=prepare_HEA,
            num_iterations=200,
            num_seeds=16,
            random_samples=20,
            result_path="results/HEA_bo_results.npz",
            y_label=r"Regret ($E^* - E$)",
        ),
        Config(
            title="OER Optimization",
            prepare_fn=partial(prepare_OER, data_path='OER/OER.csv'),
            num_iterations=200,
            num_seeds=16,
            random_samples=20,
            result_path="results/OER_bo_results.npz",
            y_label="Overpotential (mV)",
        ),
        Config(
            title="BH Reaction Optimization",
            prepare_fn=partial(prepare_BH,
                               feature_selector="random_forest",
                               min_imp=0.01, 
                               max_cum_imp=0.8),
            num_iterations=200,
            num_seeds=16,
            random_samples=20,
            result_path="results/BH_bo_results.npz",
            y_label=r"Regret ($\mathrm{yield}^* - \mathrm{yield}$)",
        ),
    ]

    results: dict[str, dict[str, np.ndarray]] = {}

    for config in configs:
        print(f"\n[Running BO for {config.title}]")
        results[config.title] = config.run(ignore_if_exists=True)

    # Plot all
    titles = [config.title for config in configs]
    ylabels = [config.y_label for config in configs]
    results_list = [results[title] for title in titles]
    plot_combined_optimization_results(
        results_list,
        titles,
        ylabels,
        fig_path="combined_bo_results.pdf"
    )
