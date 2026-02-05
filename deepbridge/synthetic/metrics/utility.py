import typing as t

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def evaluate_machine_learning_utility(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    target_column: str,
    categorical_columns: t.Optional[t.List[str]] = None,
    numerical_columns: t.Optional[t.List[str]] = None,
    problem_type: str = 'auto',
    test_size: float = 0.2,
    random_state: t.Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """
    Evaluate utility of synthetic data for machine learning tasks.

    This function trains models on both real and synthetic data and compares
    their performance on a holdout set of real data.

    Args:
        real_data: Original real dataset
        synthetic_data: Generated synthetic dataset
        target_column: Name of the target column
        categorical_columns: List of categorical feature columns
        numerical_columns: List of numerical feature columns
        problem_type: 'classification', 'regression', or 'auto' (inferred)
        test_size: Fraction of real data to use as test set
        random_state: Random seed for reproducibility
        verbose: Whether to print progress information

    Returns:
        Dictionary with utility metrics
    """
    if (
        target_column not in real_data.columns
        or target_column not in synthetic_data.columns
    ):
        raise ValueError(
            f"Target column '{target_column}' not found in datasets"
        )

    if verbose:
        print(
            f'Evaluating machine learning utility for target: {target_column}'
        )

    # Infer problem type if auto
    if problem_type == 'auto':
        if (
            real_data[target_column].dtype == 'object'
            or real_data[target_column].nunique() < 10
        ):
            problem_type = 'classification'
        else:
            problem_type = 'regression'

        if verbose:
            print(f'Inferred problem type: {problem_type}')

    # Identify feature columns
    if numerical_columns is None and categorical_columns is None:
        # Infer column types
        numerical_columns = []
        categorical_columns = []

        for col in real_data.columns:
            if col == target_column:
                continue

            if (
                pd.api.types.is_numeric_dtype(real_data[col])
                and real_data[col].nunique() > 10
            ):
                numerical_columns.append(col)
            else:
                categorical_columns.append(col)

        if verbose:
            print(
                f'Inferred {len(numerical_columns)} numerical features and {len(categorical_columns)} categorical features'
            )

    # Split real data into train and test sets
    X_real = real_data.drop(columns=[target_column])
    y_real = real_data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X_real, y_real, test_size=test_size, random_state=random_state
    )

    # Extract features and target from synthetic data
    X_synthetic = synthetic_data.drop(columns=[target_column])
    y_synthetic = synthetic_data[target_column]

    # Create preprocessing pipeline
    preprocessor = []

    if numerical_columns:
        numerical_transformer = StandardScaler()
        preprocessor.append(('num', numerical_transformer, numerical_columns))

    if categorical_columns:
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        preprocessor.append(
            ('cat', categorical_transformer, categorical_columns)
        )

    if not preprocessor:
        raise ValueError('No valid feature columns for preprocessing')

    preprocessing = ColumnTransformer(transformers=preprocessor)

    # Create and train models
    if problem_type == 'classification':
        # Convert target to categorical if needed
        if not pd.api.types.is_categorical_dtype(
            y_train
        ) and not pd.api.types.is_object_dtype(y_train):
            y_train = y_train.astype('category')
            y_test = y_test.astype('category')
            y_synthetic = y_synthetic.astype('category')

        # Classification model
        model_class = DecisionTreeClassifier
        scoring_metric = accuracy_score

        # Create pipelines
        real_model = Pipeline(
            [
                ('preprocessor', preprocessing),
                (
                    'classifier',
                    model_class(max_depth=5, random_state=random_state),
                ),
            ]
        )

        synthetic_model = Pipeline(
            [
                ('preprocessor', preprocessing),
                (
                    'classifier',
                    model_class(max_depth=5, random_state=random_state),
                ),
            ]
        )

    else:
        # Regression model
        model_class = DecisionTreeRegressor
        scoring_metric = r2_score

        # Create pipelines
        real_model = Pipeline(
            [
                ('preprocessor', preprocessing),
                (
                    'regressor',
                    model_class(max_depth=5, random_state=random_state),
                ),
            ]
        )

        synthetic_model = Pipeline(
            [
                ('preprocessor', preprocessing),
                (
                    'regressor',
                    model_class(max_depth=5, random_state=random_state),
                ),
            ]
        )

    # Train models
    if verbose:
        print('Training model on real data...')

    real_model.fit(X_train, y_train)

    if verbose:
        print('Training model on synthetic data...')

    synthetic_model.fit(X_synthetic, y_synthetic)

    # Evaluate models on test set
    if verbose:
        print('Evaluating models on test set...')

    y_pred_real = real_model.predict(X_test)
    y_pred_synthetic = synthetic_model.predict(X_test)

    # Calculate performance metrics
    real_score = scoring_metric(y_test, y_pred_real)
    synthetic_score = scoring_metric(y_test, y_pred_synthetic)

    # Calculate additional metrics for specific problem types
    if problem_type == 'classification':
        try:
            real_f1 = f1_score(y_test, y_pred_real, average='weighted')
            synthetic_f1 = f1_score(
                y_test, y_pred_synthetic, average='weighted'
            )

            additional_metrics = {
                'real_f1_score': real_f1,
                'synthetic_f1_score': synthetic_f1,
                'f1_relative_performance': synthetic_f1 / real_f1
                if real_f1 > 0
                else 0,
            }
        except:
            additional_metrics = {}
    else:
        try:
            real_rmse = np.sqrt(mean_squared_error(y_test, y_pred_real))
            synthetic_rmse = np.sqrt(
                mean_squared_error(y_test, y_pred_synthetic)
            )

            additional_metrics = {
                'real_rmse': real_rmse,
                'synthetic_rmse': synthetic_rmse,
                'rmse_relative_performance': real_rmse / synthetic_rmse
                if synthetic_rmse > 0
                else 0,
            }
        except:
            additional_metrics = {}

    # Create results dictionary
    results = {
        'problem_type': problem_type,
        'real_score': real_score,
        'synthetic_score': synthetic_score,
        'relative_performance': synthetic_score / real_score
        if real_score > 0
        else 0,
        'performance_difference': real_score - synthetic_score,
        **additional_metrics,
    }

    if verbose:
        print(f'Real data model score: {real_score:.4f}')
        print(f'Synthetic data model score: {synthetic_score:.4f}')
        print(f"Relative performance: {results['relative_performance']:.4f}")

    return results


def evaluate_statistical_fidelity(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    numerical_columns: t.Optional[t.List[str]] = None,
    categorical_columns: t.Optional[t.List[str]] = None,
    verbose: bool = False,
) -> dict:
    """
    Evaluate how well synthetic data preserves statistical properties of real data.

    This function is focused on statistical utility rather than privacy or similarity.

    Args:
        real_data: Original real dataset
        synthetic_data: Generated synthetic dataset
        numerical_columns: List of numerical column names
        categorical_columns: List of categorical column names
        verbose: Whether to print progress information

    Returns:
        Dictionary with utility metrics
    """
    from scipy import stats

    from .statistical import (
        evaluate_categorical_column,
        evaluate_numerical_column,
    )

    if verbose:
        print('Evaluating statistical fidelity...')

    # Identify numerical and categorical columns if not provided
    if numerical_columns is None and categorical_columns is None:
        numerical_columns = []
        categorical_columns = []

        for col in set(real_data.columns) & set(synthetic_data.columns):
            if (
                pd.api.types.is_numeric_dtype(real_data[col])
                and real_data[col].nunique() > 10
            ):
                numerical_columns.append(col)
            else:
                categorical_columns.append(col)

        if verbose:
            print(
                f'Inferred {len(numerical_columns)} numerical columns and {len(categorical_columns)} categorical columns'
            )

    # Initialize results dictionary
    results = {'numerical': {}, 'categorical': {}, 'overall': {}}

    # Evaluate numerical columns
    num_scores = []

    for col in numerical_columns:
        if col in real_data.columns and col in synthetic_data.columns:
            try:
                metrics = evaluate_numerical_column(
                    real_data[col], synthetic_data[col]
                )

                # Calculate a fidelity score (inverse of distance metrics)
                if 'jensen_shannon_dist' in metrics:
                    js_score = 1 - metrics['jensen_shannon_dist']
                    num_scores.append(js_score)
                    metrics['js_fidelity_score'] = js_score

                results['numerical'][col] = metrics

            except Exception as e:
                if verbose:
                    print(f'Error evaluating numerical column {col}: {str(e)}')

    # Evaluate categorical columns
    cat_scores = []

    for col in categorical_columns:
        if col in real_data.columns and col in synthetic_data.columns:
            try:
                metrics = evaluate_categorical_column(
                    real_data[col], synthetic_data[col]
                )

                # Calculate a fidelity score (inverse of distribution difference)
                if 'distribution_difference' in metrics:
                    dist_score = 1 - metrics['distribution_difference']
                    cat_scores.append(dist_score)
                    metrics['distribution_fidelity_score'] = dist_score

                results['categorical'][col] = metrics

            except Exception as e:
                if verbose:
                    print(
                        f'Error evaluating categorical column {col}: {str(e)}'
                    )

    # Calculate overall fidelity scores
    if num_scores:
        results['overall']['numerical_fidelity_score'] = sum(num_scores) / len(
            num_scores
        )

    if cat_scores:
        results['overall']['categorical_fidelity_score'] = sum(
            cat_scores
        ) / len(cat_scores)

    # Combined score (weighted average)
    if num_scores or cat_scores:
        combined_scores = num_scores + cat_scores
        results['overall']['overall_fidelity_score'] = sum(
            combined_scores
        ) / len(combined_scores)

    # Evaluate correlation fidelity if we have multiple numerical columns
    if len(numerical_columns) >= 2:
        try:
            real_corr = real_data[numerical_columns].corr().fillna(0)
            synth_corr = synthetic_data[numerical_columns].corr().fillna(0)

            # Calculate correlation difference
            corr_diff = (real_corr - synth_corr).abs().values
            avg_corr_diff = np.mean(corr_diff)

            # Convert to a fidelity score (inverse of difference)
            corr_fidelity = 1 - avg_corr_diff

            results['overall']['correlation_fidelity_score'] = corr_fidelity
        except Exception as e:
            if verbose:
                print(f'Error evaluating correlation fidelity: {str(e)}')

    if verbose:
        print('Statistical fidelity evaluation complete')
        if 'overall_fidelity_score' in results['overall']:
            print(
                f"Overall fidelity score: {results['overall']['overall_fidelity_score']:.4f}"
            )

    return results


def evaluate_query_errors(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    queries: t.List[t.Dict[str, t.Any]],
    verbose: bool = False,
) -> dict:
    """
    Evaluate accuracy of aggregate queries on synthetic data vs. real data.

    This function runs several common aggregate queries and compares the results.

    Args:
        real_data: Original real dataset
        synthetic_data: Generated synthetic dataset
        queries: List of query specifications (see examples below)
        verbose: Whether to print progress information

    Returns:
        Dictionary with query error metrics

    Example queries:
        [
            {'type': 'count', 'condition': 'age > 30'},
            {'type': 'mean', 'column': 'income', 'groupby': 'education'},
            {'type': 'sum', 'column': 'balance', 'condition': 'status == "active"'}
        ]
    """
    if verbose:
        print(f'Evaluating {len(queries)} query errors...')

    results = {'queries': [], 'overall': {}}

    absolute_errors = []
    relative_errors = []

    for i, query in enumerate(queries):
        query_type = query.get('type', 'count')
        query_name = query.get('name', f'Query_{i+1}')

        try:
            # Prepare query components
            column = query.get('column')
            condition = query.get('condition')
            groupby = query.get('groupby')

            # Apply condition if provided
            if condition:
                real_filtered = real_data.query(condition)
                synth_filtered = synthetic_data.query(condition)
            else:
                real_filtered = real_data
                synth_filtered = synthetic_data

            # Execute query based on type
            if query_type == 'count':
                if groupby:
                    real_result = real_filtered.groupby(groupby).size()
                    synth_result = synth_filtered.groupby(groupby).size()

                    # Scale synthetic counts to match real data size
                    scale_factor = (
                        len(real_data) / len(synthetic_data)
                        if len(synthetic_data) > 0
                        else 1
                    )
                    synth_result = synth_result * scale_factor
                else:
                    real_result = len(real_filtered)

                    # Scale synthetic count to match real data size
                    scale_factor = (
                        len(real_data) / len(synthetic_data)
                        if len(synthetic_data) > 0
                        else 1
                    )
                    synth_result = len(synth_filtered) * scale_factor

            elif query_type == 'mean':
                if not column:
                    raise ValueError(
                        f"Column must be specified for query type '{query_type}'"
                    )

                if groupby:
                    real_result = real_filtered.groupby(groupby)[column].mean()
                    synth_result = synth_filtered.groupby(groupby)[
                        column
                    ].mean()
                else:
                    real_result = real_filtered[column].mean()
                    synth_result = synth_filtered[column].mean()

            elif query_type == 'sum':
                if not column:
                    raise ValueError(
                        f"Column must be specified for query type '{query_type}'"
                    )

                if groupby:
                    real_result = real_filtered.groupby(groupby)[column].sum()
                    synth_result = synth_filtered.groupby(groupby)[
                        column
                    ].sum()

                    # Scale synthetic sums to match real data size
                    scale_factor = (
                        len(real_data) / len(synthetic_data)
                        if len(synthetic_data) > 0
                        else 1
                    )
                    synth_result = synth_result * scale_factor
                else:
                    real_result = real_filtered[column].sum()

                    # Scale synthetic sum to match real data size
                    scale_factor = (
                        len(real_data) / len(synthetic_data)
                        if len(synthetic_data) > 0
                        else 1
                    )
                    synth_result = synth_filtered[column].sum() * scale_factor

            elif query_type == 'median':
                if not column:
                    raise ValueError(
                        f"Column must be specified for query type '{query_type}'"
                    )

                if groupby:
                    real_result = real_filtered.groupby(groupby)[
                        column
                    ].median()
                    synth_result = synth_filtered.groupby(groupby)[
                        column
                    ].median()
                else:
                    real_result = real_filtered[column].median()
                    synth_result = synth_filtered[column].median()

            elif query_type == 'std':
                if not column:
                    raise ValueError(
                        f"Column must be specified for query type '{query_type}'"
                    )

                if groupby:
                    real_result = real_filtered.groupby(groupby)[column].std()
                    synth_result = synth_filtered.groupby(groupby)[
                        column
                    ].std()
                else:
                    real_result = real_filtered[column].std()
                    synth_result = synth_filtered[column].std()

            else:
                raise ValueError(f'Unsupported query type: {query_type}')

            # Calculate errors
            if groupby:
                # Align indices for grouped results
                combined = pd.DataFrame(
                    {'real': real_result, 'synthetic': synth_result}
                ).fillna(0)

                absolute_error = np.abs(
                    combined['real'] - combined['synthetic']
                ).mean()

                # Avoid division by zero
                nonzero_mask = combined['real'] != 0
                if nonzero_mask.any():
                    relative_error = np.abs(
                        (
                            combined.loc[nonzero_mask, 'real']
                            - combined.loc[nonzero_mask, 'synthetic']
                        )
                        / combined.loc[nonzero_mask, 'real']
                    ).mean()
                else:
                    relative_error = np.nan
            else:
                absolute_error = np.abs(real_result - synth_result)
                relative_error = (
                    np.abs(real_result - synth_result) / np.abs(real_result)
                    if np.abs(real_result) > 0
                    else np.nan
                )

            # Store query results
            query_result = {
                'name': query_name,
                'type': query_type,
                'column': column,
                'groupby': groupby,
                'condition': condition,
                'absolute_error': float(absolute_error),
                'relative_error': float(relative_error)
                if not np.isnan(relative_error)
                else None,
            }

            results['queries'].append(query_result)

            # Add to overall error calculations
            if not np.isnan(absolute_error):
                absolute_errors.append(absolute_error)

            if not np.isnan(relative_error):
                relative_errors.append(relative_error)

            if verbose:
                print(
                    f'  - {query_name}: Absolute Error = {absolute_error:.4f}, Relative Error = {relative_error:.4f}'
                    if not np.isnan(relative_error)
                    else f'  - {query_name}: Absolute Error = {absolute_error:.4f}'
                )

        except Exception as e:
            if verbose:
                print(f'Error evaluating query {query_name}: {str(e)}')

            # Add failed query to results
            query_result = {
                'name': query_name,
                'type': query_type,
                'error': str(e),
            }
            results['queries'].append(query_result)

    # Calculate overall metrics
    if absolute_errors:
        results['overall']['mean_absolute_error'] = float(
            np.mean(absolute_errors)
        )
        results['overall']['max_absolute_error'] = float(
            np.max(absolute_errors)
        )

    if relative_errors:
        results['overall']['mean_relative_error'] = float(
            np.mean(relative_errors)
        )
        results['overall']['max_relative_error'] = float(
            np.max(relative_errors)
        )

    if verbose:
        if 'mean_relative_error' in results['overall']:
            print(
                f"Overall mean relative error: {results['overall']['mean_relative_error']:.4f}"
            )

    return results
