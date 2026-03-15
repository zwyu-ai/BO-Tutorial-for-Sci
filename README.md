# Bayesian Optimisation Tutorial for Scientific Discovery

This repository accompanies a tutorial on Bayesian Optimisation (BO) for experimental and computational scientists. It shows how BO can be used to formalise the classic scientific cycle of hypothesise–experiment–update by:

- Framing scientific discovery problems as optimisation tasks over a design space
- Building probabilistic surrogate models (Gaussian processes and related models)
- Designing acquisition functions to select informative, high‑value experiments
- Demonstrating end‑to‑end BO workflows on real scientific case studies

The material is tiered so that different readers can enter at the right level:

- **Experimentalists / practitioners** – run ready‑made BO experiments on real datasets and compare HEBO, basic BO and random search
- **Method / ML researchers** – study GP‑based surrogates, acquisition functions, and full BO loops implemented from scratch
- **General scientific audience** – gain intuition for uncertainty‑aware, data‑efficient experiment design

## Simple Bayesian Optimisation Workflow

At a high level, BO wraps an expensive black‑box evaluator (an experiment or simulation) with a probabilistic model that guides where to sample next:

1. **Define the problem**  
   Specify a design space (what you can change) and an objective evaluator (how to score a given design).

2. **Collect initial data**  
   Evaluate the objective at a small set of initial points to form a dataset of inputs and outputs.

3. **Fit a surrogate model**  
   Train a probabilistic model (often a Gaussian process) on all collected data to obtain a predictive mean and uncertainty over the design space.

4. **Build an acquisition function**  
   Use the surrogate’s predictions and uncertainties to construct an acquisition function (e.g. UCB, EI, PI) that scores how promising each candidate experiment is, balancing exploration (high uncertainty) and exploitation (high predicted value).

5. **Choose the next experiment**  
   Maximise the acquisition function over the design space to select the next point(s) to evaluate.

6. **Evaluate and update**  
   Run the real experiment or simulation at the chosen point(s), append the new observations to the dataset, and refit or update the surrogate.

7. **Repeat until stopping**  
   Iterate steps 3–6 until a budget, convergence, or practical stopping criterion is met, then recommend the best design seen (or inferred) as the final solution.

Many practical extensions build on this basic loop: batched/parallel selection of multiple experiments per round, contextual BO, handling heteroscedastic noise, and human‑in‑the‑loop decision making.

## Repository Structure and Key Files

This section points you to the most relevant files depending on what you are looking for.

### 1. End‑to‑End BO Experiments on Real Scientific Problems

- [experiment.py](experiment.py)  
  Main entry point to reproduce the benchmark experiments comparing:
  - HEBO (Heteroscedastic Evolutionary Bayesian Optimisation)
  - Basic BO with Lower Confidence Bound (LCB) acquisition
  - Random search

  It sets up multiple scientific tasks and runs BO loops on each:
  - **HER optimisation** – hydrogen evolution reaction (electrocatalysis)
  - **HEA nanozyme optimisation** – high‑entropy alloy catalysts
  - **OER optimisation** – oxygen evolution reaction
  - **BH reaction optimisation** – organic synthesis (Buchwald–Hartwig coupling)
   - **Molecule optimisation** – molecular design over a SMILES library (QED objective). See `molecule.ipynb` and `examples/Molecule/utils.py` for a worked example.

  The script produces per‑task result files (e.g. `results/BH_bo_results.npz`) and a combined comparison plot saved as `results/combined_bo_results.pdf`.

- [utils.py](utils.py)  
  Shared utilities for running and visualising BO:
  - `run_optimization` – runs HEBO, basic BO (LCB), and random search on a given design space and oracle
  - `plot_regret` – plots convergence (regret / best‑so‑far curves) for a single task
  - `plot_combined_optimization_results` – side‑by‑side comparison across multiple tasks

- [examples/](examples)  
  Case‑study‑specific code and data:
  - [examples/HER](examples/HER): utilities and virtual data for HER optimisation
  - [examples/HEA](examples/HEA): BO setup, oracle, and results for HEA nanozyme optimisation
  - [examples/OER](examples/OER): OER dataset and utilities
  - [examples/BH](examples/BH): BH reaction dataset and BO utilities

These modules expose `create_problem_*` functions (e.g. `create_problem_HER`, `create_problem_HEA`) that return a HEBO `DesignSpace` and an oracle function, and are imported by `experiment.py`.

### 2. BO with HEBO on a Synthetic Function

- [HEBO_tutorial.ipynb](HEBO_tutorial.ipynb)  
  A standalone notebook showing how to use the HEBO package on a 2D synthetic test function (Branin–Hoo):
  - Defines a rich **design space** with numeric, integer, categorical, logarithmic, and other parameter types
  - Implements the Branin–Hoo objective as a Pandas‑based oracle
  - Runs and compares:
    - Basic BO (LCB)
    - Sequential HEBO
    - Batched/parallel HEBO
  - Visualises regret curves and highlights the effect of batch vs sequential BO

This is a good starting point if you want to see HEBO “out of the box” before looking at the full experimental benchmarks.

- [molecule.ipynb](molecule.ipynb)  
   Notebook demonstrating molecular optimisation over a SMILES library: it shows how to
   define a categorical design space of molecules (SMILES), use an RDKit‑based QED oracle,
   and run HEBO / basic BO vs random search. The helper utilities live in [examples/Molecule](examples/Molecule).

### 3. BO Building Blocks: Gaussian Processes and Acquisition Functions

For readers interested in the mathematical and implementation details of BO’s core components, the `coding_illustrations/` folder provides minimal, didactic implementations.

- [coding_illustrations/gp_bo_gpytorch.ipynb](coding_illustrations/gp_bo_gpytorch.ipynb)  
  Shows how to implement BO using **Gpytorch**:
  - Constructs an exact GP model and trains it on 1D synthetic data
  - Implements acquisition functions (UCB, EI, PI) using the GP posterior
  - Runs a full BO loop in 2D on a synthetic objective, selecting new points via UCB on a grid
  - Visualises GP predictions, uncertainty bands, and acquisition landscapes

- [coding_illustrations/gp_bo_sklearn.ipynb](coding_illustrations/gp_bo_sklearn.ipynb)  
  Mirrors the same concepts using **scikit‑learn**:
  - Builds a GaussianProcessRegressor with composite kernels
  - Implements UCB, EI, and PI acquisition functions
  - Executes a complete BO loop on a 2D synthetic function

These notebooks are ideal if you want to understand “what BO is doing under the hood” rather than just calling a high‑level optimiser.

### 4. Datasets for Scientific Discovery

Simple datasets have been included to create objective oracles and specify optimisation problems for scientific discovery.

- [examples/BH/BH_dataset.csv](examples/BH/BH_dataset.csv) – Buchwald–Hartwig reaction dataset
- [examples/HER/HER_virtual_data.csv](examples/HER/HER_virtual_data.csv) – virtual HER data
- [examples/OER/OER.csv](examples/OER/OER.csv) and [examples/OER/OER_clean.csv](examples/OER/OER_clean.csv) – OER datasets
- [examples/HEA/data](examples/HEA/data) – HEA nanozyme data used by the HEA oracle
 - [examples/Molecule/zinc.txt.gz](examples/Molecule/zinc.txt.gz) – compressed ZINC SMILES list used by the Molecule example

### 5. Results

- [results/](results)  
  Stores BO outcomes produced by `experiment.py`, such as:
  - `*_bo_results.npz` – NumPy archives containing the HEBO / BO (LCB) / random search histories
  - `combined_bo_results.pdf` – comparison figure across all case studies

You can safely delete these files if you want to rerun all experiments from scratch; they will be regenerated.

## Usage

### Installation and Dependencies

This project targets Python 3.9+ and relies on standard scientific Python packages plus HEBO and (for some notebooks) PyTorch, Gpytorch, scikit‑learn, and AutoGluon.

1. (Recommended) Create and activate a virtual environment. We recommend using `uv`:
   
   ```bash
   uv venv --python 3.XX
   .venv/Scripts/activate  # activate the virtual environment
   ```

2. Install base dependencies:

   ```bash
   uv pip install -r requirements.txt
   ```
3. Install HEBO:
   
   ```bash
   uv pip install hebo
   ```
  
  Some may encounter issues when installing one of the dependencies `Gpy` that is required by HEBO but not necessarily needed in HEBO. To bypass, add `--no-deps`.

### How to Reproduce the Case Studies

1. Install the dependencies as described above.
2. From the repository root, run:

   ```bash
   python experiment.py
   ```

   This will sequentially run BO on all configured tasks (HER, HEA, OER, BH). If result files already exist, the script will skip recomputation unless you change the configuration.

3. Inspect the saved results in the `results/` directory and the generated combined comparison figure.

For deeper exploration, open the notebooks in `HEBO_tutorial.ipynb` and `coding_illustrations/` and execute them cell‑by‑cell.

**Note for Molecule Optimisation:**
The repository also contains a Molecule example (molecular optimisation) in `molecule.ipynb` and `examples/Molecule`. This example is intentionally provided as a standalone demo: it defines a categorical design space over SMILES strings and uses an RDKit‑based QED oracle. Treating SMILES as raw categorical strings requires custom feature/descriptor layers (mapping molecules to numeric representations) before a GP surrogate can be applied; for this reason the Molecule example is not included in the main `experiment.py` runner and is aimed at experienced users who want to extend the tutorial. Advanced users can adapt the example by computing molecular descriptors or fingerprints and converting the design space to numeric features suitable for HEBO or other GP‑based surrogates. The Molecule utilities use RDKit (declared in `requirements.txt`) and a ZINC SMILES archive at `examples/Molecule/zinc.txt.gz`.

### Who Is This Repository For?

- **Experimental scientists** interested in more principled, automated experiment design
- **Data scientists / ML practitioners** working on BO for applied scientific problems
- **Method developers** exploring new BO variants or surrogates for scientific discovery

The examples are deliberately designed to be readable and easily modified so you can plug in your own design spaces, datasets, and objectives.

### Citing This Tutorial

If you use this material in your research, teaching, or software, please cite the associated Bayesian Optimisation tutorial (TO BE RELEASED).

