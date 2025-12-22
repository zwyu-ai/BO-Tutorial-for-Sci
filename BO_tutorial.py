from sklearn.model_selection import train_test_split
import warnings

from utils import run_optimization, plot_combined_optimization_results
from OER.utils import get_preprocessed_data as get_preprocessed_data_OER
from OER.utils import get_design_space as get_design_space_OER
from OER.utils import construct_oracle as construct_oracle_OER
from HER.utils import get_preprocessed_data as get_preprocessed_data_HER
from HER.utils import get_design_space as get_design_space_HER
from HER.utils import construct_oracle as construct_oracle_HER
from BH.utils import get_preprocessed_data as get_preprocessed_data_BH
from BH.utils import get_design_space as get_design_space_BH
from BH.utils import construct_oracle as construct_oracle_BH

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    NUM_ITERATIONS = 100
    NUM_SEEDS = 16
    RANDOM_SAMPLES = 10

    # OER
    OER_data, categorical_feat, numerical_feat, target_col = get_preprocessed_data_OER(data_path="OER/OER_clean.csv")
    OER_design_space = get_design_space_OER(data_df=OER_data, categorical_features=categorical_feat)
    # Prepare Data for Oracle
    X = OER_data[categorical_feat + numerical_feat].copy()
    y = OER_data[target_col].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    OER_oracle = construct_oracle_OER(
        X_train, y_train,
        categorical_feat,
        impl="random_forest",
        validate=True
    )
    hebo_history, bo_history, rs_history = run_optimization(
        space=OER_design_space,
        oracle=OER_oracle,
        num_iterations=NUM_ITERATIONS,
        random_seeds=list(range(42, 42 + NUM_SEEDS)),
        random_samples=RANDOM_SAMPLES
    )
    OER_results = {
        'HEBO': hebo_history,
        'BO (LCB)': bo_history,
        'Random Search': rs_history
    }

    # HER
    HER_data = get_preprocessed_data_HER(data_path='HER/HER_virtual_data.csv')
    HER_space = get_design_space_HER()
    HER_oracle = construct_oracle_HER(data_df=HER_data, impl="random_forest")
    hebo_history, bo_history, rs_history = run_optimization(
        space=HER_space,
        oracle=HER_oracle,
        num_iterations=NUM_ITERATIONS,
        random_seeds=list(range(42, 42 + NUM_SEEDS)),
        random_samples=RANDOM_SAMPLES
    )
    HER_results = {
        'HEBO': hebo_history,
        'BO (LCB)': bo_history,
        'Random Search': rs_history
    }

    # BH
    BH_data = get_preprocessed_data_BH(data_path='BH/BH_dataset.csv')
    BH_space = get_design_space_BH(data_df=BH_data)
    BH_oracle = construct_oracle_BH(data_df=BH_data, impl="random_forest")
    hebo_history, bo_history, rs_history = run_optimization(
        space=BH_space,
        oracle=BH_oracle,
        num_iterations=NUM_ITERATIONS,
        random_seeds=list(range(42, 42 + NUM_SEEDS)),
        random_samples=RANDOM_SAMPLES
    )
    BH_results = {
        'HEBO': hebo_history,
        'BO (LCB)': bo_history,
        'Random Search': rs_history
    }

    # Plot all
    results_list = [OER_results, HER_results, BH_results]
    titles = [
        "OER Catalyst Optimization",
        "HER Optimization",
        "BH Optimization"
    ]
    ylabels = [
        "Overpotential (mV)",
        "Regret ($\mathrm{yield}^* - \mathrm{yield}$)",
        "Regret ($\mathrm{yield}^* - \mathrm{yield}$)"
    ]
    plot_combined_optimization_results(
        results_list,
        titles,
        ylabels,
        fig_path="combined_bo_results.pdf"
    )
