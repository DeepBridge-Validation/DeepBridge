"""
Fairness testing suite for machine learning models.

This module provides a comprehensive suite for evaluating fairness of ML models,
following industry standards and regulatory requirements (EEOC, ECOA, Fair Lending Act).

Integrates seamlessly with the DeepBridge experiment framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
import time

from deepbridge.validation.fairness.metrics import FairnessMetrics


class FairnessSuite:
    """
    Complete fairness testing suite integrated with DeepBridge.

    Tests model fairness across protected attributes (e.g., race, gender, age)
    using industry-standard metrics and regulatory compliance checks.

    Usage:
    ------
    >>> from deepbridge.core.db_data import DBDataset
    >>> from deepbridge.validation.wrappers.fairness_suite import FairnessSuite
    >>>
    >>> # Create dataset
    >>> dataset = DBDataset(features=X, target=y, model=model)
    >>>
    >>> # Initialize fairness suite
    >>> fairness = FairnessSuite(
    ...     dataset=dataset,
    ...     protected_attributes=['gender', 'race'],
    ...     verbose=True
    ... )
    >>>
    >>> # Run tests
    >>> results = fairness.config('full').run()
    >>>
    >>> # Check results
    >>> print(f"Fairness Score: {results['overall_fairness_score']:.3f}")
    >>> print(f"Critical Issues: {len(results['critical_issues'])}")
    """

    # Configuration templates
    _CONFIG_TEMPLATES = {
        'quick': {
            'metrics': ['statistical_parity', 'disparate_impact'],
            'include_pretrain': False,
            'include_threshold_analysis': False,
            'include_confusion_matrix': False,
            'description': 'Quick fairness check with 2 core metrics'
        },
        'medium': {
            'metrics': [
                'statistical_parity', 'equal_opportunity', 'disparate_impact',
                'precision_difference', 'accuracy_difference'
            ],
            'include_pretrain': True,
            'include_threshold_analysis': False,
            'include_confusion_matrix': True,
            'description': 'Standard fairness assessment with 5 post-training + 4 pre-training metrics'
        },
        'full': {
            'metrics': [
                # All 11 post-training metrics
                'statistical_parity', 'equal_opportunity', 'equalized_odds',
                'disparate_impact', 'false_negative_rate_difference',
                'conditional_acceptance', 'conditional_rejection',
                'precision_difference', 'accuracy_difference',
                'treatment_equality', 'entropy_index'
            ],
            'include_pretrain': True,
            'include_threshold_analysis': True,
            'include_confusion_matrix': True,
            'description': 'Comprehensive fairness analysis with all 15 metrics + threshold analysis'
        }
    }

    # Pre-training metrics (model-independent)
    _PRETRAIN_METRICS = ['class_balance', 'concept_balance', 'kl_divergence', 'js_divergence']

    # Post-training metrics (model-dependent)
    _POSTTRAIN_METRICS = [
        'statistical_parity', 'equal_opportunity', 'equalized_odds', 'disparate_impact',
        'false_negative_rate_difference', 'conditional_acceptance', 'conditional_rejection',
        'precision_difference', 'accuracy_difference', 'treatment_equality', 'entropy_index'
    ]

    def __init__(self,
                 dataset,
                 protected_attributes: List[str],
                 privileged_groups: Optional[Dict[str, Any]] = None,
                 verbose: bool = False,
                 age_grouping: bool = True,
                 age_grouping_strategy: str = 'median',
                 age_threshold: Optional[float] = None):
        """
        Initialize fairness testing suite.

        Parameters:
        -----------
        dataset : DBDataset
            Dataset object from DeepBridge containing features, target, and model
        protected_attributes : List[str]
            List of protected attributes to test for fairness
            Examples: ['gender', 'race', 'age', 'ethnicity']
        privileged_groups : Dict[str, Any], optional
            Definition of privileged groups for each protected attribute
            Example: {'gender': 'male', 'race': 'white', 'age': 'young'}
            If None, automatically determined as group with highest positive rate
        verbose : bool, default=False
            Whether to print detailed progress information
        age_grouping : bool, default=True
            Whether to automatically group age variables. Age variables are detected
            by common naming patterns (e.g., 'age', 'idade', 'vl_idd_aa')
        age_grouping_strategy : str, default='median'
            Strategy for grouping age variables:
            - 'median': Binary split at median age (default: 45 years)
            - 'adea': ADEA framework for employment (<40, 40-49, 50-59, 60+)
            - 'ecoa': ECOA framework for credit (18-29, 30-39, 40-49, 50-59, 60+)
        age_threshold : float, optional
            Custom threshold for 'median' strategy. If None, uses data median.

        Raises:
        -------
        ValueError
            If protected_attributes not found in dataset features
        """
        self.dataset = dataset
        self.protected_attributes = protected_attributes
        self.privileged_groups = privileged_groups or {}
        self.verbose = verbose
        self.age_grouping = age_grouping
        self.age_grouping_strategy = age_grouping_strategy
        self.age_threshold = age_threshold

        # Initialize metrics calculator
        self.metrics_calculator = FairnessMetrics()

        # Store current configuration
        self.current_config = None
        self.current_config_name = None

        # Store age grouping metadata
        self.age_grouping_applied = {}  # Will store which attributes were age-grouped

        # Validate that protected attributes exist in dataset
        X = self.dataset.get_feature_data()
        missing_attrs = [attr for attr in protected_attributes if attr not in X.columns]

        if missing_attrs:
            raise ValueError(
                f"Protected attributes not found in dataset: {missing_attrs}\n"
                f"Available features: {list(X.columns)}"
            )

        if self.verbose:
            print(f"FairnessSuite initialized")
            print(f"  Protected attributes: {protected_attributes}")
            print(f"  Dataset shape: {X.shape}")
            if self.age_grouping:
                print(f"  Age grouping: Enabled (strategy: {age_grouping_strategy})")

    def config(self, config_name: str = 'full') -> 'FairnessSuite':
        """
        Set configuration for fairness tests.

        Parameters:
        -----------
        config_name : str, default='full'
            Configuration level: 'quick', 'medium', or 'full'

        Returns:
        --------
        self : FairnessSuite
            Returns self for method chaining

        Example:
        --------
        >>> fairness.config('full').run()
        """
        if config_name not in self._CONFIG_TEMPLATES:
            raise ValueError(
                f"Unknown configuration: {config_name}. "
                f"Available options: {list(self._CONFIG_TEMPLATES.keys())}"
            )

        self.current_config = self._CONFIG_TEMPLATES[config_name].copy()
        self.current_config_name = config_name

        if self.verbose:
            print(f"\nConfigured for {config_name} fairness testing")
            print(f"  Description: {self.current_config['description']}")
            print(f"  Metrics: {self.current_config['metrics']}")

        return self

    def _is_age_column(self, column_name: str, data: pd.Series) -> bool:
        """
        Detect if a column represents age based on name and data characteristics.

        Parameters:
        -----------
        column_name : str
            Name of the column
        data : pd.Series
            Column data

        Returns:
        --------
        bool : True if column is likely an age variable
        """
        # Common age column name patterns (case-insensitive)
        age_patterns = [
            'age', 'idade', 'edad', 'anni', 'ans',  # age in various languages
            'vl_idd', 'vl_idade',  # common Brazilian patterns
            'age_', '_age', '_idd_'  # suffixes/prefixes
        ]

        # Check if column name matches patterns
        col_lower = column_name.lower()
        name_matches = any(pattern in col_lower for pattern in age_patterns)

        if not name_matches:
            return False

        # Verify data characteristics (age should be numeric with reasonable range)
        if not pd.api.types.is_numeric_dtype(data):
            return False

        # Check if values are in typical age range (0-120)
        min_val = data.min()
        max_val = data.max()

        if min_val < 0 or max_val > 120:
            return False

        # Check if range is reasonable for age (at least 10 years span)
        if max_val - min_val < 10:
            return False

        return True

    def _create_age_groups(
        self,
        age_data: pd.Series,
        strategy: str = 'median',
        threshold: Optional[float] = None
    ) -> pd.Series:
        """
        Create age groups based on specified strategy.

        Parameters:
        -----------
        age_data : pd.Series
            Age values to group
        strategy : str
            Grouping strategy: 'median', 'adea', or 'ecoa'
        threshold : float, optional
            Custom threshold for 'median' strategy

        Returns:
        --------
        pd.Series : Grouped age labels

        Raises:
        -------
        ValueError
            If strategy is unknown
        """
        age_groups = pd.Series(index=age_data.index, dtype=str)

        if strategy == 'median':
            # Binary split at median (or custom threshold)
            if threshold is None:
                threshold = age_data.median()

            age_groups[age_data >= threshold] = f"Grupo A (>= {threshold:.0f} anos)"
            age_groups[age_data < threshold] = f"Grupo B (< {threshold:.0f} anos)"

            if self.verbose:
                count_a = (age_data >= threshold).sum()
                count_b = (age_data < threshold).sum()
                pct_a = (count_a / len(age_data)) * 100
                pct_b = (count_b / len(age_data)) * 100
                print(f"    Median strategy (threshold={threshold:.0f}):")
                print(f"      Grupo A (>= {threshold:.0f}): {count_a} ({pct_a:.1f}%)")
                print(f"      Grupo B (< {threshold:.0f}): {count_b} ({pct_b:.1f}%)")

        elif strategy == 'adea':
            # ADEA framework: <40, 40-49, 50-59, 60+
            age_groups[age_data < 40] = "< 40 anos"
            age_groups[(age_data >= 40) & (age_data < 50)] = "40-49 anos"
            age_groups[(age_data >= 50) & (age_data < 60)] = "50-59 anos"
            age_groups[age_data >= 60] = "60+ anos"

            if self.verbose:
                print(f"    ADEA strategy (employment framework):")
                for group in ["< 40 anos", "40-49 anos", "50-59 anos", "60+ anos"]:
                    count = (age_groups == group).sum()
                    pct = (count / len(age_data)) * 100
                    print(f"      {group}: {count} ({pct:.1f}%)")

        elif strategy == 'ecoa':
            # ECOA framework: 18-29, 30-39, 40-49, 50-59, 60+
            age_groups[age_data < 30] = "18-29 anos"
            age_groups[(age_data >= 30) & (age_data < 40)] = "30-39 anos"
            age_groups[(age_data >= 40) & (age_data < 50)] = "40-49 anos"
            age_groups[(age_data >= 50) & (age_data < 60)] = "50-59 anos"
            age_groups[age_data >= 60] = "60+ anos"

            if self.verbose:
                print(f"    ECOA strategy (credit framework):")
                for group in ["18-29 anos", "30-39 anos", "40-49 anos", "50-59 anos", "60+ anos"]:
                    count = (age_groups == group).sum()
                    pct = (count / len(age_data)) * 100
                    print(f"      {group}: {count} ({pct:.1f}%)")

        else:
            raise ValueError(
                f"Unknown age grouping strategy: {strategy}. "
                f"Valid options: 'median', 'adea', 'ecoa'"
            )

        return age_groups

    def _preprocess_protected_attributes(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess protected attributes, applying age grouping if enabled.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature data containing protected attributes

        Returns:
        --------
        pd.DataFrame : Preprocessed data with age columns grouped
        """
        X_processed = X.copy()

        if not self.age_grouping:
            return X_processed

        # Check each protected attribute for age columns
        for attr in self.protected_attributes:
            if attr not in X_processed.columns:
                continue

            # Check if this is an age column
            if self._is_age_column(attr, X_processed[attr]):
                if self.verbose:
                    print(f"\n  📅 Age variable detected: {attr}")

                # Create age groups
                age_groups = self._create_age_groups(
                    X_processed[attr],
                    strategy=self.age_grouping_strategy,
                    threshold=self.age_threshold
                )

                # Replace original column with grouped version
                X_processed[attr] = age_groups

                # Store metadata
                self.age_grouping_applied[attr] = {
                    'strategy': self.age_grouping_strategy,
                    'threshold': self.age_threshold,
                    'original_range': (
                        float(X[attr].min()),
                        float(X[attr].max())
                    ),
                    'groups': age_groups.unique().tolist()
                }

        return X_processed

    def _calculate_confusion_matrix_by_group(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_feature: np.ndarray
    ) -> Dict[str, Dict[str, int]]:
        """
        Calculate detailed confusion matrix for each group.

        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        sensitive_feature : array-like
            Protected attribute values

        Returns:
        --------
        dict : Confusion matrix per group
            {
                'Group_A': {'TP': int, 'FP': int, 'TN': int, 'FN': int, 'total': int},
                'Group_B': {'TP': int, 'FP': int, 'TN': int, 'FN': int, 'total': int}
            }
        """
        from sklearn.metrics import confusion_matrix

        groups = np.unique(sensitive_feature)
        cm_by_group = {}

        for group in groups:
            mask = sensitive_feature == group
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]

            if len(y_true_group) == 0:
                cm_by_group[str(group)] = {
                    'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'total': 0
                }
                continue

            # Calculate confusion matrix
            cm = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1])

            cm_by_group[str(group)] = {
                'TN': int(cm[0, 0]),  # True Negatives
                'FP': int(cm[0, 1]),  # False Positives
                'FN': int(cm[1, 0]),  # False Negatives
                'TP': int(cm[1, 1]),  # True Positives
                'total': int(len(y_true_group))
            }

        return cm_by_group

    def run_threshold_analysis(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        sensitive_feature: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
        optimize_for: str = 'fairness'
    ) -> Dict[str, Any]:
        """
        Analyze how fairness metrics vary with different classification thresholds.

        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred_proba : array-like
            Predicted probabilities (for positive class)
        sensitive_feature : array-like
            Protected attribute values
        thresholds : array-like, optional
            Thresholds to test (default: 0.01 to 0.99 in steps of 0.01)
        optimize_for : str, default='fairness'
            Optimization criterion: 'fairness', 'f1', or 'balanced'

        Returns:
        --------
        dict : Threshold analysis results
            {
                'optimal_threshold': float,
                'optimal_metrics': Dict,
                'threshold_curve': List[Dict],  # Metrics at each threshold
                'recommendations': List[str]
            }
        """
        if thresholds is None:
            thresholds = np.arange(0.01, 1.0, 0.01)

        from sklearn.metrics import f1_score

        results = []

        for threshold in thresholds:
            # Generate predictions at this threshold
            y_pred = (y_pred_proba >= threshold).astype(int)

            # Calculate key fairness metrics
            sp = self.metrics_calculator.statistical_parity(y_pred, sensitive_feature)
            di = self.metrics_calculator.disparate_impact(y_pred, sensitive_feature)

            # Calculate performance metric (F1)
            try:
                f1 = f1_score(y_true, y_pred, zero_division=0)
            except:
                f1 = 0.0

            # Store results
            results.append({
                'threshold': float(threshold),
                'statistical_parity_ratio': sp['ratio'],
                'disparate_impact_ratio': di['ratio'],
                'f1_score': float(f1),
                'passes_80_rule': di['passes_threshold']
            })

        # Convert to DataFrame for easier analysis
        import pandas as pd
        df_results = pd.DataFrame(results)

        # Find optimal threshold based on criterion
        if optimize_for == 'fairness':
            # Maximize disparate impact ratio (closest to 1.0)
            df_results['fairness_score'] = 1 - np.abs(df_results['disparate_impact_ratio'] - 1.0)
            optimal_idx = df_results['fairness_score'].idxmax()
        elif optimize_for == 'f1':
            # Maximize F1 score
            optimal_idx = df_results['f1_score'].idxmax()
        else:  # balanced
            # Balanced: maximize product of F1 and fairness
            df_results['fairness_score'] = 1 - np.abs(df_results['disparate_impact_ratio'] - 1.0)
            df_results['balanced_score'] = df_results['f1_score'] * df_results['fairness_score']
            optimal_idx = df_results['balanced_score'].idxmax()

        optimal_threshold = df_results.loc[optimal_idx, 'threshold']
        optimal_metrics = df_results.loc[optimal_idx].to_dict()

        # Generate recommendations
        recommendations = []
        current_default = df_results[df_results['threshold'] == 0.5].iloc[0]

        if optimal_threshold != 0.5:
            recommendations.append(
                f"Considere alterar threshold de 0.5 para {optimal_threshold:.3f} "
                f"para melhor {optimize_for}"
            )

        if current_default['disparate_impact_ratio'] < 0.8:
            recommendations.append(
                "⚠️ Default threshold (0.5) violates EEOC 80% rule - "
                "adjustment required"
            )

        if optimal_metrics['disparate_impact_ratio'] < 0.8:
            recommendations.append(
                "🚨 CRITICAL: Even with optimal threshold, model doesn't achieve compliance - "
                "consider retraining the model"
            )

        return {
            'optimal_threshold': float(optimal_threshold),
            'optimal_metrics': optimal_metrics,
            'threshold_curve': df_results.to_dict('records'),
            'recommendations': recommendations,
            'default_threshold_metrics': current_default.to_dict()
        }

    def _calculate_metric(
        self,
        metric_name: str,
        y_true: Optional[np.ndarray],
        y_pred: Optional[np.ndarray],
        sensitive_feature: np.ndarray
    ) -> Dict[str, Any]:
        """
        Helper to calculate any fairness metric dynamically.

        Parameters:
        -----------
        metric_name : str
            Name of the metric to calculate
        y_true : array-like, optional
            True labels (not needed for some metrics)
        y_pred : array-like, optional
            Predicted labels (not needed for pre-training metrics)
        sensitive_feature : array-like
            Protected attribute values

        Returns:
        --------
        dict : Metric results
        """
        # Pre-training metrics (don't need predictions)
        if metric_name == 'class_balance':
            return self.metrics_calculator.class_balance(y_true, sensitive_feature)
        elif metric_name == 'concept_balance':
            return self.metrics_calculator.concept_balance(y_true, sensitive_feature)
        elif metric_name == 'kl_divergence':
            return self.metrics_calculator.kl_divergence(y_true, sensitive_feature)
        elif metric_name == 'js_divergence':
            return self.metrics_calculator.js_divergence(y_true, sensitive_feature)

        # Post-training metrics (need predictions)
        elif metric_name == 'statistical_parity':
            return self.metrics_calculator.statistical_parity(y_pred, sensitive_feature)
        elif metric_name == 'equal_opportunity':
            return self.metrics_calculator.equal_opportunity(y_true, y_pred, sensitive_feature)
        elif metric_name == 'equalized_odds':
            return self.metrics_calculator.equalized_odds(y_true, y_pred, sensitive_feature)
        elif metric_name == 'disparate_impact':
            return self.metrics_calculator.disparate_impact(y_pred, sensitive_feature)
        elif metric_name == 'false_negative_rate_difference':
            return self.metrics_calculator.false_negative_rate_difference(y_true, y_pred, sensitive_feature)
        elif metric_name == 'conditional_acceptance':
            return self.metrics_calculator.conditional_acceptance(y_true, y_pred, sensitive_feature)
        elif metric_name == 'conditional_rejection':
            return self.metrics_calculator.conditional_rejection(y_true, y_pred, sensitive_feature)
        elif metric_name == 'precision_difference':
            return self.metrics_calculator.precision_difference(y_true, y_pred, sensitive_feature)
        elif metric_name == 'accuracy_difference':
            return self.metrics_calculator.accuracy_difference(y_true, y_pred, sensitive_feature)
        elif metric_name == 'treatment_equality':
            return self.metrics_calculator.treatment_equality(y_true, y_pred, sensitive_feature)
        elif metric_name == 'entropy_index':
            return self.metrics_calculator.entropy_index(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    def _collect_dataset_info(self, X: pd.DataFrame, y_true: np.ndarray) -> Dict[str, Any]:
        """
        Collect comprehensive dataset information for reporting.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature data containing protected attributes
        y_true : np.ndarray
            True labels

        Returns:
        --------
        dict : Dataset information including sample size, target distribution,
               and protected attributes distributions
        """
        import pandas as pd

        info = {
            'total_samples': len(X),
            'target_distribution': {},
            'protected_attributes_distribution': {}
        }

        # Target distribution
        target_counts = pd.Series(y_true).value_counts().sort_index()
        total = len(y_true)

        for value, count in target_counts.items():
            info['target_distribution'][str(value)] = {
                'count': int(count),
                'percentage': float(count / total * 100)
            }

        # Protected attributes distribution
        for attr in self.protected_attributes:
            attr_counts = X[attr].value_counts()
            attr_info = {
                'unique_values': int(X[attr].nunique()),
                'distribution': {}
            }

            for value, count in attr_counts.items():
                attr_info['distribution'][str(value)] = {
                    'count': int(count),
                    'percentage': float(count / total * 100)
                }

            info['protected_attributes_distribution'][attr] = attr_info

        return info

    def run(self) -> Dict[str, Any]:
        """
        Execute fairness tests.

        Returns:
        --------
        dict : Comprehensive fairness test results
            {
                'protected_attributes': List[str],
                'pretrain_metrics': Dict[str, Dict],  # Pre-training metrics
                'posttrain_metrics': Dict[str, Dict],  # Post-training metrics
                'confusion_matrix': Dict[str, Dict],  # Confusion matrix by group (if enabled)
                'threshold_analysis': Dict,  # Threshold analysis (if enabled)
                'overall_fairness_score': float,  # 0-1, higher is better
                'warnings': List[str],
                'critical_issues': List[str],
                'summary': Dict,
                'config': Dict,
                'dataset_info': Dict  # NEW: Dataset information for reporting
            }

        Raises:
        -------
        ValueError
            If no model found in dataset or no configuration set
        """
        # Set default config if none set
        if self.current_config is None:
            if self.verbose:
                print("No configuration set, using 'full' configuration")
            self.config('full')

        start_time = time.time()

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"RUNNING FAIRNESS TESTS - {self.current_config_name.upper()}")
            print(f"{'='*70}")

        # Get data
        X = self.dataset.get_feature_data()
        y_true = self.dataset.get_target_data()
        y_pred = None
        y_pred_proba = None

        # Preprocess protected attributes (apply age grouping if enabled)
        if self.verbose and self.age_grouping:
            print(f"\n🔧 PREPROCESSING PROTECTED ATTRIBUTES")
        X = self._preprocess_protected_attributes(X)

        # Get predictions for post-training metrics
        needs_predictions = any(m in self._POSTTRAIN_METRICS for m in self.current_config['metrics'])

        if needs_predictions:
            # Try to use existing predictions first
            if hasattr(self.dataset, 'train_predictions') and self.dataset.train_predictions is not None:
                if self.verbose:
                    print("Using existing predictions from dataset...")
                # Use pre-computed predictions
                train_preds = self.dataset.train_predictions

                # Handle different formats: DataFrame or dict
                if isinstance(train_preds, dict):
                    # Dictionary format: {'y_pred': array, 'y_proba': array}
                    if 'y_pred' in train_preds:
                        y_pred = train_preds['y_pred']
                    if 'y_proba' in train_preds:
                        y_pred_proba = train_preds['y_proba']
                        # If y_proba is 2D, take second column (class 1)
                        if len(y_pred_proba.shape) == 2 and y_pred_proba.shape[1] > 1:
                            y_pred_proba = y_pred_proba[:, 1]
                elif hasattr(train_preds, 'columns'):
                    # DataFrame format
                    if 'prediction' in train_preds.columns:
                        y_pred = train_preds['prediction'].values
                        # Try to get probabilities
                        if 'proba_class_1' in train_preds.columns:
                            y_pred_proba = train_preds['proba_class_1'].values
                        elif 'probability' in train_preds.columns:
                            y_pred_proba = train_preds['probability'].values
            # If no pre-computed predictions, generate them
            elif hasattr(self.dataset, 'model') and self.dataset.model is not None:
                if self.verbose:
                    print("Generating predictions from model...")

                # Get only the columns the model was trained on
                # Exclude protected attributes if they weren't used in training
                X_for_prediction = X.copy()

                # Try to filter to only features the model expects
                if hasattr(self.dataset.model, 'feature_names_in_'):
                    model_features = self.dataset.model.feature_names_in_
                    # Keep only columns that the model was trained on
                    available_features = [f for f in model_features if f in X.columns]
                    X_for_prediction = X[available_features]
                    if self.verbose and len(available_features) < len(X.columns):
                        excluded = set(X.columns) - set(available_features)
                        print(f"  Excluding {len(excluded)} features from prediction: {excluded}")

                y_pred = self.dataset.model.predict(X_for_prediction)

                # Try to get probabilities for threshold analysis
                if hasattr(self.dataset.model, 'predict_proba'):
                    y_pred_proba = self.dataset.model.predict_proba(X_for_prediction)[:, 1]
                elif hasattr(self.dataset.model, 'decision_function'):
                    # For models with decision_function (like SVM)
                    from sklearn.preprocessing import MinMaxScaler
                    decision = self.dataset.model.decision_function(X_for_prediction)
                    scaler = MinMaxScaler()
                    y_pred_proba = scaler.fit_transform(decision.reshape(-1, 1)).flatten()
            else:
                raise ValueError(
                    "No model found in dataset. Cannot compute predictions.\n"
                    "Provide a trained model when creating DBDataset."
                )

        # Collect dataset information for reporting
        dataset_info = self._collect_dataset_info(X, y_true)

        # Initialize results
        results = {
            'protected_attributes': self.protected_attributes,
            'pretrain_metrics': {},
            'posttrain_metrics': {},
            'confusion_matrix': {},
            'threshold_analysis': None,
            'overall_fairness_score': 0.0,
            'warnings': [],
            'critical_issues': [],
            'age_grouping_applied': self.age_grouping_applied,
            'dataset_info': dataset_info,
            'config': {
                'name': self.current_config_name,
                'metrics_tested': self.current_config['metrics'],
                'include_pretrain': self.current_config.get('include_pretrain', False),
                'include_confusion_matrix': self.current_config.get('include_confusion_matrix', False),
                'include_threshold_analysis': self.current_config.get('include_threshold_analysis', False),
                'age_grouping': self.age_grouping,
                'age_grouping_strategy': self.age_grouping_strategy if self.age_grouping else None
            }
        }

        # 1. PRE-TRAINING METRICS (if enabled)
        if self.current_config.get('include_pretrain', False):
            if self.verbose:
                print(f"\n📋 PRE-TRAINING ANALYSIS (Model-Independent)")

            for attr in self.protected_attributes:
                if self.verbose:
                    print(f"\n  Analyzing: {attr}")

                sensitive_feature = X[attr].values
                attr_pretrain = {}

                for metric_name in self._PRETRAIN_METRICS:
                    if self.verbose:
                        print(f"    ✓ {metric_name}")

                    result = self._calculate_metric(metric_name, y_true, None, sensitive_feature)
                    attr_pretrain[metric_name] = result

                    # Check interpretation for warnings
                    interp = result.get('interpretation', '')
                    if '✗ Red' in interp or 'CRITICAL' in interp.upper():
                        critical = f"{attr} [{metric_name}]: {interp}"
                        results['critical_issues'].append(critical)
                        if self.verbose:
                            print(f"        🚨 {interp}")
                    elif '⚠ Yellow' in interp or 'MODERATE' in interp.upper():
                        warning = f"{attr} [{metric_name}]: {interp}"
                        results['warnings'].append(warning)
                        if self.verbose:
                            print(f"        ⚠️  {interp}")

                results['pretrain_metrics'][attr] = attr_pretrain

        # 2. POST-TRAINING METRICS
        if self.verbose:
            print(f"\n📊 POST-TRAINING ANALYSIS (Model-Dependent)")

        for attr in self.protected_attributes:
            if self.verbose:
                print(f"\n  Testing fairness for: {attr}")

            sensitive_feature = X[attr].values
            attr_posttrain = {}

            # Run each configured post-training metric
            for metric_name in self.current_config['metrics']:
                if metric_name not in self._POSTTRAIN_METRICS:
                    continue  # Skip if not a post-training metric

                if self.verbose:
                    print(f"    ✓ {metric_name}")

                result = self._calculate_metric(metric_name, y_true, y_pred, sensitive_feature)
                attr_posttrain[metric_name] = result

                # Check interpretation for warnings
                interp = result.get('interpretation', '')
                if '✗ Red' in interp or 'CRITICAL' in interp.upper():
                    critical = f"{attr} [{metric_name}]: {interp}"
                    results['critical_issues'].append(critical)
                    if self.verbose:
                        print(f"        🚨 {interp}")
                elif '⚠ Yellow' in interp or 'MODERATE' in interp.upper():
                    warning = f"{attr} [{metric_name}]: {interp}"
                    results['warnings'].append(warning)
                    if self.verbose:
                        print(f"        ⚠️  {interp}")

                # Additional checks for specific metrics
                if metric_name == 'disparate_impact':
                    if not result.get('passes_threshold', True):
                        critical = (
                            f"{attr}: Disparate Impact CRITICAL "
                            f"(ratio={result['ratio']:.3f} < 0.8) - LEGAL RISK"
                        )
                        if critical not in results['critical_issues']:
                            results['critical_issues'].append(critical)

            results['posttrain_metrics'][attr] = attr_posttrain

        # 3. CONFUSION MATRIX ANALYSIS (if enabled)
        if self.current_config.get('include_confusion_matrix', False) and y_pred is not None:
            if self.verbose:
                print(f"\n🔢 CONFUSION MATRIX ANALYSIS")

            for attr in self.protected_attributes:
                sensitive_feature = X[attr].values
                cm = self._calculate_confusion_matrix_by_group(y_true, y_pred, sensitive_feature)
                results['confusion_matrix'][attr] = cm

                if self.verbose:
                    print(f"\n  {attr}:")
                    for group, matrix in cm.items():
                        print(f"    {group}: TP={matrix['TP']}, FP={matrix['FP']}, "
                              f"TN={matrix['TN']}, FN={matrix['FN']} (total={matrix['total']})")

        # 4. THRESHOLD ANALYSIS (if enabled)
        if (self.current_config.get('include_threshold_analysis', False) and
            y_pred_proba is not None and len(self.protected_attributes) > 0):

            if self.verbose:
                print(f"\n📈 THRESHOLD ANALYSIS")

            # Use first protected attribute for threshold analysis
            attr = self.protected_attributes[0]
            sensitive_feature = X[attr].values

            threshold_results = self.run_threshold_analysis(
                y_true, y_pred_proba, sensitive_feature, optimize_for='balanced'
            )
            results['threshold_analysis'] = threshold_results

            if self.verbose:
                print(f"  Optimal threshold: {threshold_results['optimal_threshold']:.3f}")
                print(f"  Recommendations:")
                for rec in threshold_results['recommendations']:
                    print(f"    • {rec}")

        # 5. CALCULATE OVERALL FAIRNESS SCORE
        results['overall_fairness_score'] = self._calculate_fairness_score_v2(
            results['pretrain_metrics'],
            results['posttrain_metrics']
        )

        # 6. GENERATE SUMMARY
        results['summary'] = {
            'total_attributes_tested': len(self.protected_attributes),
            'pretrain_metrics_count': len(results['pretrain_metrics']),
            'posttrain_metrics_count': len(results['posttrain_metrics']),
            'attributes_with_warnings': len(set(
                w.split(':')[0].split('[')[0].strip() for w in results['warnings']
            )) if results['warnings'] else 0,
            'critical_issues_found': len(results['critical_issues']),
            'overall_assessment': self._assess_fairness(
                results['overall_fairness_score']
            ),
            'execution_time': time.time() - start_time
        }

        # Print final summary
        if self.verbose:
            self._print_summary(results)

        # Return FairnessResult object instead of dict for consistency
        # This enables automatic HTML report generation via .save_html()
        from deepbridge.core.experiment.results import FairnessResult

        return FairnessResult(results=results, metadata={
            'test_type': 'fairness',
            'execution_time': results['summary']['execution_time']
        })

    def _calculate_fairness_score_v2(
        self,
        pretrain_metrics: Dict,
        posttrain_metrics: Dict
    ) -> float:
        """
        Calculate overall fairness score (0-1, higher is better) considering all metrics.

        Uses weighted average with higher weights for legally critical metrics:
        - Disparate Impact: 30% (most critical - EEOC compliance)
        - Statistical Parity: 20%
        - Equal Opportunity: 15%
        - Equalized Odds: 15%
        - Pre-training metrics: 10% combined
        - Other post-training: 10% combined

        Parameters:
        -----------
        pretrain_metrics : Dict
            Pre-training metrics results per attribute
        posttrain_metrics : Dict
            Post-training metrics results per attribute

        Returns:
        --------
        float : Overall fairness score (0-1)
        """
        scores = []
        weights = []

        # Process post-training metrics
        for attr, attr_metrics in posttrain_metrics.items():
            # Disparate Impact (weight: 0.30 - most critical)
            if 'disparate_impact' in attr_metrics:
                di_score = attr_metrics['disparate_impact']['ratio']
                scores.append(di_score)
                weights.append(0.30)

            # Statistical Parity (weight: 0.20)
            if 'statistical_parity' in attr_metrics:
                sp_score = attr_metrics['statistical_parity']['ratio']
                scores.append(sp_score)
                weights.append(0.20)

            # Equal Opportunity (weight: 0.15)
            if 'equal_opportunity' in attr_metrics:
                eo_disparity = attr_metrics['equal_opportunity']['disparity']
                eo_score = max(0, 1 - eo_disparity)
                scores.append(eo_score)
                weights.append(0.15)

            # Equalized Odds (weight: 0.15)
            if 'equalized_odds' in attr_metrics:
                eq_disparity = attr_metrics['equalized_odds']['combined_disparity']
                eq_score = max(0, 1 - eq_disparity)
                scores.append(eq_score)
                weights.append(0.15)

            # Other post-training metrics (weight: 0.02 each, up to 0.10 total)
            other_metrics = [
                'false_negative_rate_difference', 'conditional_acceptance',
                'conditional_rejection', 'precision_difference',
                'accuracy_difference'
            ]
            for metric_name in other_metrics:
                if metric_name in attr_metrics:
                    result = attr_metrics[metric_name]
                    # Convert value to score (closer to 0 is better)
                    value = abs(result.get('value', 0))
                    score = max(0, 1 - value)
                    scores.append(score)
                    weights.append(0.02)

        # Process pre-training metrics (weight: 0.025 each, up to 0.10 total)
        for attr, attr_metrics in pretrain_metrics.items():
            for metric_name in self._PRETRAIN_METRICS:
                if metric_name in attr_metrics:
                    result = attr_metrics[metric_name]

                    # Convert to score based on metric type
                    if metric_name == 'class_balance':
                        value = abs(result.get('value', 0))
                        score = max(0, 1 - value)
                    elif metric_name == 'concept_balance':
                        value = abs(result.get('value', 0))
                        score = max(0, 1 - (value / 0.5))  # Normalize to 0.5 max disparity
                    elif metric_name in ['kl_divergence', 'js_divergence']:
                        value = result.get('value', 0)
                        score = max(0, 1 - value)  # KL/JS should be close to 0
                    else:
                        score = 0.5  # Default middle score

                    scores.append(score)
                    weights.append(0.025)

        if not scores:
            return 0.0

        # Weighted average
        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _calculate_fairness_score(self, metrics: Dict) -> float:
        """
        Legacy method for backward compatibility.
        Delegates to _calculate_fairness_score_v2.
        """
        return self._calculate_fairness_score_v2({}, metrics)

    def _assess_fairness(self, score: float) -> str:
        """Assess fairness level based on score"""
        if score >= 0.95:
            return "EXCELLENT - Highly fair and compliant model"
        elif score >= 0.85:
            return "GOOD - Adequate fairness for production"
        elif score >= 0.70:
            return "MODERATE - Requires attention and possible remediation"
        else:
            return "CRITICAL - Intervention required before deployment"

    def _print_summary(self, results: Dict):
        """Print formatted summary of results"""
        print(f"\n{'='*70}")
        print(f"FAIRNESS ASSESSMENT SUMMARY")
        print(f"{'='*70}")
        print(f"Overall Fairness Score: {results['overall_fairness_score']:.3f} / 1.000")
        print(f"Assessment: {results['summary']['overall_assessment']}")
        print(f"\nConfiguration: {results['config']['name']}")
        print(f"Attributes Tested: {results['summary']['total_attributes_tested']}")

        # Show metrics breakdown
        if results['pretrain_metrics']:
            print(f"Pre-training Metrics: {len(self._PRETRAIN_METRICS)} metrics × "
                  f"{results['summary']['pretrain_metrics_count']} attributes")
        if results['posttrain_metrics']:
            total_posttrain = sum(
                len(metrics) for metrics in results['posttrain_metrics'].values()
            )
            print(f"Post-training Metrics: {total_posttrain} total calculations")

        # Show confusion matrix info
        if results['confusion_matrix']:
            print(f"Confusion Matrix: ✓ Generated for {len(results['confusion_matrix'])} attributes")

        # Show threshold analysis info
        if results['threshold_analysis']:
            ta = results['threshold_analysis']
            print(f"Threshold Analysis: ✓ Optimal = {ta['optimal_threshold']:.3f}")

        print(f"\nIssues Found:")
        print(f"  Critical Issues: {results['summary']['critical_issues_found']}")
        print(f"  Warnings: {len(results['warnings'])}")
        print(f"  Attributes with Issues: {results['summary']['attributes_with_warnings']}")

        print(f"\nExecution Time: {results['summary']['execution_time']:.2f}s")

        # Print critical issues
        if results['critical_issues']:
            print(f"\n🚨 CRITICAL ISSUES ({len(results['critical_issues'])}):")
            for issue in results['critical_issues'][:10]:  # Show first 10
                print(f"   • {issue}")
            if len(results['critical_issues']) > 10:
                print(f"   ... and {len(results['critical_issues']) - 10} more")

        # Print warnings
        if results['warnings']:
            print(f"\n⚠️  WARNINGS ({len(results['warnings'])}):")
            for warning in results['warnings'][:5]:  # Show first 5
                print(f"   • {warning}")
            if len(results['warnings']) > 5:
                print(f"   ... and {len(results['warnings']) - 5} more")

        # Print threshold recommendations
        if results.get('threshold_analysis') and results['threshold_analysis'].get('recommendations'):
            print(f"\n💡 THRESHOLD RECOMMENDATIONS:")
            for rec in results['threshold_analysis']['recommendations']:
                print(f"   • {rec}")

        print(f"{'='*70}\n")

    def get_detailed_results(self, attribute: str) -> Optional[Dict]:
        """
        Get detailed results for a specific protected attribute.

        Parameters:
        -----------
        attribute : str
            Protected attribute name

        Returns:
        --------
        dict or None : Detailed metrics for the attribute
        """
        # Must run tests first
        # This would typically be called after run()
        pass  # Implementation would retrieve from stored results
