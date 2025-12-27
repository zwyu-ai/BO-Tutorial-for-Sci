import argparse
from pathlib import Path

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
    result_path: Path
    num_iterations: int = 100
    num_seeds: int = 1
    random_samples: int = 10
    y_label: str = "y"
    optimizer: str = "all"
    opt_keys = ['HEBO', 'BO (LCB)', 'Random Search']
    res_names = ['hebo.npz', 'bo.npz', 'rs.npz']

    def __post_init__(self):
        if self.optimizer == 'hebo':
            self.opt_keys = ['HEBO']
            self.res_names = ['hebo.npz']
        elif self.optimizer == 'bo':
            self.opt_keys = ['BO (LCB)']
            self.res_names = ['bo.npz']
        elif self.optimizer == 'rs':
            self.opt_keys = ['Random Search']
            self.res_names = ['rs.npz']

    def run(self, ignore_if_exists: bool = False):
        design_space, oracle = self.prepare_fn()
        results = {}

        print("=" * 70)
        print(self.title)
        print("=" * 70)

        for key, filename in zip(self.opt_keys, self.res_names):
            if ignore_if_exists and (self.result_path / filename).exists():
                res = np.load(self.result_path / filename, allow_pickle=True)['arr_0']
                print(f"✓ {self.result_path / filename} exists, skipping...")

            else:
                res = run_optimization(
                    type=key,
                    space=design_space,
                    oracle=oracle,
                    num_iterations=self.num_iterations,
                    random_seeds=list(range(0, self.num_seeds)),
                    random_samples=self.random_samples
                )

                os.makedirs(self.result_path, exist_ok=True)
                np.savez(self.result_path / filename, res)

            results[key] = res

        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, choices=['all', 'hebo', 'bo', 'rs'], default='all',
                        help="Choice of optimizer, defaults to `all`")
    parser.add_argument('--dataset', type=str, choices=['all', 'her', 'hea', 'oer', 'bh'], default='all',
                        help="Choice of dataset on which to run experiment, defaults to `all`")
    args = parser.parse_args()

    configs = {
        "her": Config(
            title="HER Optimization",
            prepare_fn=prepare_HER,
            num_iterations=300,
            num_seeds=16,
            random_samples=15,
            result_path=Path("HER/results"),
            y_label=r"Regret ($\mathrm{yield}^* - \mathrm{yield}$)",
            optimizer=args.optimizer,
        ),
        "hea": Config(
            title="HEA Nanozyme Optimization",
            prepare_fn=prepare_HEA,
            num_iterations=300,
            num_seeds=16,
            random_samples=15,
            result_path=Path("HEA/results"),
            y_label=r"Regret ($E^* - E$)",
            optimizer=args.optimizer,
        ),
        "oer": Config(
            title="OER Optimization",
            prepare_fn=partial(prepare_OER, data_path='OER/OER_clean.csv'),
            num_iterations=300,
            num_seeds=16,
            random_samples=15,
            result_path=Path("OER/results"),
            y_label="Overpotential (mV)",
            optimizer=args.optimizer,
        ),
        "bh": Config(
            title="BH Reaction Optimization",
            prepare_fn=prepare_BH,
            num_iterations=300,
            num_seeds=16,
            random_samples=15,
            result_path=Path("BH/results"),
            y_label=r"Regret ($\mathrm{yield}^* - \mathrm{yield}$)",
            optimizer=args.optimizer,
        ),
    }
    if args.dataset != 'all':
        configs = {args.dataset: configs[args.dataset]}

    results: dict[str, dict[str, np.ndarray]] = {}

    for key, config in configs.items():
        print(f"\n[Running BO for {config.title}]")
        results[config.title] = config.run(ignore_if_exists=True)

    # Plot all
    if args.dataset == 'all':
        titles = [config.title for config in configs.values()]
        ylabels = [config.y_label for config in configs.values()]
        results_list = [results[title] for title in titles]
        plot_combined_optimization_results(
            results_list,
            titles,
            ylabels,
            fig_path="combined_bo_results.pdf"
        )
