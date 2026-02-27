import tarfile
import pandas as pd
import numpy as np
from typing import Literal
from functools import partial
from rdkit import Chem
from rdkit.Chem import QED
from hebo.design_space.design_space import DesignSpace


def load_zinc_smiles(zinc_file="examples/Molecule/zinc.txt.gz"):
    with tarfile.open(zinc_file, "r:gz") as tar:
        if (member := tar.extractfile("zinc.txt")) is not None:
            content = member.read().decode("utf-8").splitlines()
            return content
        else:
            raise FileNotFoundError(f"Error: zinc.txt not found in the archive")


def qed_objective(smiles: str) -> float:
    """
    QED objective function for molecular optimization.
    Returns QED score (0-1) – maximize this value.
    Penalizes invalid SMILES with a score of 0.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0  # Invalid SMILES = worst possible score
    return QED.qed(mol)


type OracleImpl = Literal["qed"]


def oracle(xs: pd.DataFrame,
           impl: OracleImpl = "qed") -> np.ndarray:
    if impl == "qed":
        return np.array([qed_objective(smiles) for smiles in xs["SMILES"]]).reshape(-1, 1)
    else:
        raise ValueError(f"Unsupported oracle implementation: {impl}")


def create_problem(
    zinc_file: str = "examples/Molecule/zinc.txt.gz",
    oracle_impl: OracleImpl = "qed"
):
    smiles_list = load_zinc_smiles(zinc_file)
    oracle_fn = partial(oracle, impl=oracle_impl)
    design_space = DesignSpace().parse([
        {'name': 'SMILES', 'type': 'cat', 'categories': smiles_list}
    ])

    return design_space, oracle_fn
