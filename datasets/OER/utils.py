from typing import Callable

import numpy as np
import pandas as pd
from hebo.design_space.design_space import DesignSpace


def get_preprocessed_data(data_path: str = 'OER/OER_clean.csv') -> tuple[pd.DataFrame, list[str], list[str], str]:
    """
    Returns
        clean pd.DataFrame
        list of categorical feature names
        list of numerical feature names
    """
    df = pd.read_csv(data_path)

    # 1. Target Column Processing
    target_col = 'Overpotential mV @10 mA cm-2'

    # Drop rows with missing target
    df_clean = df.dropna(subset=[target_col]).copy()

    # 2. Remove duplicate rows
    df_clean = df_clean.drop_duplicates()

    # 3. Handle comma-separated numbers (e.g., "2,280" -> 2280)
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            try:
                df_clean[col] = df_clean[col].astype(str).str.replace(',', '').str.replace('"', '')
            except:
                pass

    # 4. Select Features
    categorical_features = ['Metal_1', 'Metal_2', 'Metal_3']
    numerical_features = [
        'Metal_1_Proportion', 'Metal_2_Proportion', 'Metal_3_Proportion',
        'Hydrothermal Temp degree', 'Hydrothermal Time min',
        'Annealing Temp degree', 'Annealing Time min',
        'Proton Concentration M', 'Catalyst_Loading mg cm -2'
    ]

    # 5. Clean Categorical Features
    for col in categorical_features:
        df_clean[col] = df_clean[col].fillna('None')
        df_clean[col] = df_clean[col].astype(str).str.strip()
        # Standardize 'None' values
        df_clean[col] = df_clean[col].replace(['nan', 'NaN', 'NA', ''], 'None')

    # 6. Clean Numerical Features column-specific
    for col in numerical_features:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean[col] = df_clean[col].fillna(0)
    for col in ['Hydrothermal Temp degree', 'Hydrothermal Time min',
                'Annealing Temp degree', 'Annealing Time min',
                'Proton Concentration M', 'Catalyst_Loading mg cm -2']:
        # Remove outliers using IQR method
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

    # 7. Clean target column
    df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce')
    df_clean = df_clean.dropna(subset=[target_col])

    # 8. Remove extreme outliers in target
    Q1 = df_clean[target_col].quantile(0.05)
    Q3 = df_clean[target_col].quantile(0.95)
    df_clean = df_clean[(df_clean[target_col] >= Q1) & (df_clean[target_col] <= Q3)]

    return df_clean, categorical_features, numerical_features, target_col


def get_design_space(data_df: pd.DataFrame, categorical_features: list[str]) -> DesignSpace:
    # Define Design Space for Optimization
    params = []

    # Categorical Parameters
    for col in categorical_features:
        unique_vals = data_df[col].unique().tolist()
        if 'None' not in unique_vals:
            unique_vals.append('None')
        params.append({'name': col, 'type': 'cat', 'categories': unique_vals})

    # Numerical Parameters
    for col in ['Metal_1_Proportion', 'Metal_2_Proportion', 'Metal_3_Proportion']:
        min_val = float(0.0)
        max_val = float(100.0)
        params.append({'name': col, 'type': 'num', 'lb': min_val, 'ub': max_val})
    for col in ['Hydrothermal Temp degree', 'Hydrothermal Time min',
                'Annealing Temp degree', 'Annealing Time min']:
        min_val = float(data_df[col].min())
        max_val = float(data_df[col].max())
        if min_val == max_val:
            max_val += 1.0
        params.append({'name': col, 'type': 'int', 'lb': min_val, 'ub': max_val})
    for col in ['Proton Concentration M', 'Catalyst_Loading mg cm -2']:
        min_val = float(data_df[col].min())
        max_val = float(data_df[col].max())
        if min_val == max_val:
            max_val += 1.0
        params.append({'name': col, 'type': 'num', 'lb': min_val, 'ub': max_val})

    space = DesignSpace().parse(params)
    return space


def construct_oracle(
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        cat_cols: list,
        impl: str = "random_forest",
        validate: bool = True
) -> Callable[[pd.DataFrame], np.ndarray]:
    """
    Constructs a machine learning oracle for predicting target values.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : np.ndarray
        Training targets
    cat_cols : list
        List of categorical column names
    impl : str
        Implementation type ('random_forest', 'gradient_boosting')
    validate : bool
        Whether to perform cross-validation

    Returns:
    --------
    oracle : Callable
        Function that predicts target values for new inputs
    """

    # Encode categorical features
    X_encoded = pd.get_dummies(X_train, columns=cat_cols)
    train_cols = X_encoded.columns.tolist()

    # Train model
    if impl == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    elif impl == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown implementation: {impl}")

    model.fit(X_encoded, y_train.ravel())

    # Cross-validation
    if validate:
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(
            model, X_encoded, y_train.ravel(),
            cv=5, scoring='r2', n_jobs=-1
        )
        print(f"\n✓ {impl} Oracle:")
        print(f"  Cross-validation R² scores: {cv_scores}")
        print(f"  Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': train_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"\n  Top 10 important features:")
        print(importance_df.head(10).to_string(index=False))

    # Define oracle function
    def oracle_func(x_df):
        x_enc = pd.get_dummies(x_df, columns=cat_cols)
        for c in train_cols:
            if c not in x_enc.columns:
                x_enc[c] = 0
        x_enc = x_enc[train_cols]
        return model.predict(x_enc).reshape(-1, 1)

    return oracle_func


def prepare(data_path: str = "OER/OER_clean.csv", 
            oracle_impl: str = "random_forest"):
    from sklearn.model_selection import train_test_split

    data_df, categorical_feat, numerical_feat, target_col = get_preprocessed_data(data_path=data_path)
    design_space = get_design_space(data_df=data_df, categorical_features=categorical_feat)
    # Prepare Data for Oracle
    X = data_df[categorical_feat + numerical_feat].copy()
    y = np.asarray(data_df[target_col].values).reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    oracle = construct_oracle(
        X_train, y_train,
        categorical_feat,
        impl=oracle_impl,
        validate=True
    )
    return design_space, oracle