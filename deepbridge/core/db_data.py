import typing as t
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from deepbridge.utils.data_validator import DataValidator
from deepbridge.utils.feature_manager import FeatureManager
from deepbridge.utils.model_handler import ModelHandler
from deepbridge.utils.dataset_formatter import DatasetFormatter
from deepbridge.utils.synthetic_data import SyntheticDataGenerator

class DBDataset:
    """
    DBDataset wraps training and test datasets along with optional model and predictions.
    """
    
    def __init__(
        self,
        data: t.Optional[t.Union[pd.DataFrame, t.Any]] = None,
        train_data: t.Optional[t.Union[pd.DataFrame, t.Any]] = None,
        test_data: t.Optional[t.Union[pd.DataFrame, t.Any]] = None,
        target_column: t.Optional[str] = None,
        features: t.Optional[t.List[str]] = None,
        model_path: t.Optional[t.Union[str, Path]] = None,
        model: t.Optional[t.Any] = None,  # Parameter for directly passing a model object
        train_predictions: t.Optional[pd.DataFrame] = None,
        test_predictions: t.Optional[pd.DataFrame] = None,
        prob_cols: t.Optional[t.List[str]] = None,
        categorical_features: t.Optional[t.List[str]] = None,
        max_categories: t.Optional[int] = None,
        dataset_name: t.Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = None,
        synthetic: bool = False,
        synthetic_sample: t.Optional[int] = None
    ):
        # Initialize helper classes
        self._validator = DataValidator()
        self._model_handler = ModelHandler()
        
        # Validate input data
        self._validator.validate_data_input(data, train_data, test_data, target_column)
        
        # Validate that only one of model_path, model, or prob_cols is provided
        params_provided = sum(param is not None for param in [model_path, model, prob_cols])
        if params_provided > 1:
            raise ValueError("You must provide only one of 'model_path', 'model', or 'prob_cols' parameters. "
                         "Provide the model directly if it's already loaded in the session, "
                         "or model_path if you need to load it from a file. "
                         "If you don't have the model, provide the probabilities.")
        
        # Validate that if synthetic=True, either model_path or model must be provided
        if synthetic and model_path is None and model is None:
            raise ValueError("It is not possible to create synthetic data from probabilities. "
                         "You must provide either 'model_path' or 'model' when 'synthetic' is True.")
        
        # Process and store data
        if data is not None:
            self._process_unified_data(data, target_column, features, prob_cols, test_size)
        else:
            self._process_split_data(train_data, test_data, target_column, features, prob_cols)
        
        self._target_column = target_column
        self._dataset_name = dataset_name
        self._random_state = random_state
        
        # Handle synthetic data parameters
        self._synthetic = synthetic
        
        # Calculate default synthetic_sample size if not provided
        if synthetic_sample is None:
            if data is not None:
                self._synthetic_sample = int(len(data) * 0.1)
            else:
                self._synthetic_sample = int((len(train_data) + len(test_data)) * 0.1)
        else:
            self._synthetic_sample = synthetic_sample
        
        # Initialize feature manager and process features
        self._feature_manager = FeatureManager(self._data, self._features)
        self._categorical_features = (
            self._feature_manager.infer_categorical_features(max_categories)
            if categorical_features is None
            else self._validate_categorical_features(categorical_features)
        )
        
        # Handle model, model_path, or probabilities (only one of them should be provided at this point)
        if model_path is not None:
            # Load model from path
            self._model_handler.load_model(
                model_path,
                features=self._features,
                data={'train': self._train_data, 'test': self._test_data}
            )
        elif model is not None:
            # Use the model that's already loaded
            self._model_handler.model = model
            
            # Generate predictions if data is available
            if self._train_data is not None or self._test_data is not None:
                try:
                    # Create dictionary with available data
                    data_dict = {}
                    if self._train_data is not None:
                        data_dict['train'] = self._train_data
                    if self._test_data is not None:
                        data_dict['test'] = self._test_data
                    
                    # Generate predictions using the provided model
                    self._model_handler.generate_predictions(
                        data_dict,
                        self._features
                    )
                except Exception as e:
                    print(f"Warning: Could not generate predictions using the provided model: {str(e)}")
        elif prob_cols is not None:
            # Initialize model handler with predictions
            self._model_handler.set_predictions(
                self._train_data,
                self._test_data,
                train_predictions,
                test_predictions,
                prob_cols
            )
        
        self._formatter = DatasetFormatter(
            dataset_name=dataset_name,
            feature_manager=self._feature_manager,
            model_handler=self._model_handler,
            target_column=self._target_column
        )
        
        # Generate synthetic data if requested
        if self._synthetic:
            self._generate_synthetic_data()

    def _process_unified_data(
        self,
        data: t.Union[pd.DataFrame, t.Any],
        target_column: str,
        features: t.List[str],
        prob_cols: t.List[str],
        test_size: float
    ) -> None:
        """Process unified dataset."""
        # Convert scikit-learn Bunch or other types to DataFrame if needed
        if not isinstance(data, pd.DataFrame):
            # Try to convert from scikit-learn dataset
            try:
                # Handle scikit-learn Bunch object
                if hasattr(data, 'data') and hasattr(data, 'target'):
                    if features is None and hasattr(data, 'feature_names'):
                        features = list(data.feature_names)
                    
                    # Create DataFrame from data and target
                    feature_data = pd.DataFrame(data.data, columns=features)
                    target_data = pd.Series(data.target, name=target_column)
                    data = pd.concat([feature_data, target_data], axis=1)
                else:
                    # Try to convert any other object to DataFrame
                    data = pd.DataFrame(data)
            except Exception as e:
                raise ValueError(f"Could not convert input data to DataFrame: {str(e)}")
        
        # Now check if target column exists
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        self._features = self._validator.validate_features(features, data, target_column, prob_cols)
        
        self._original_data = data.copy()
        self._data = data[self._features].copy()
        
        train_idx = int(len(data) * (1 - test_size))
        self._train_data = data.iloc[:train_idx].copy()
        self._test_data = data.iloc[train_idx:].copy()

    def _process_split_data(
        self,
        train_data: t.Union[pd.DataFrame, t.Any],
        test_data: t.Union[pd.DataFrame, t.Any],
        target_column: str,
        features: t.Optional[t.List[str]],
        prob_cols: t.Optional[t.List[str]]
    ) -> None:
        """Process split train/test datasets."""
        # Convert scikit-learn datasets to DataFrames if needed
        if not isinstance(train_data, pd.DataFrame):
            try:
                # Handle scikit-learn Bunch object
                if hasattr(train_data, 'data') and hasattr(train_data, 'target'):
                    if features is None and hasattr(train_data, 'feature_names'):
                        features = list(train_data.feature_names)
                    
                    # Create DataFrame from data and target
                    train_feature_data = pd.DataFrame(train_data.data, columns=features)
                    train_target_data = pd.Series(train_data.target, name=target_column)
                    train_data = pd.concat([train_feature_data, train_target_data], axis=1)
                else:
                    # Try to convert any other object to DataFrame
                    train_data = pd.DataFrame(train_data)
            except Exception as e:
                raise ValueError(f"Could not convert training data to DataFrame: {str(e)}")
                
        if not isinstance(test_data, pd.DataFrame):
            try:
                # Handle scikit-learn Bunch object
                if hasattr(test_data, 'data') and hasattr(test_data, 'target'):
                    if features is None and hasattr(test_data, 'feature_names'):
                        features = list(test_data.feature_names)
                    
                    # Create DataFrame from data and target
                    test_feature_data = pd.DataFrame(test_data.data, columns=features)
                    test_target_data = pd.Series(test_data.target, name=target_column)
                    test_data = pd.concat([test_feature_data, test_target_data], axis=1)
                else:
                    # Try to convert any other object to DataFrame
                    test_data = pd.DataFrame(test_data)
            except Exception as e:
                raise ValueError(f"Could not convert test data to DataFrame: {str(e)}")
        
        if len(train_data) == 0 or len(test_data) == 0:
            raise ValueError("Training and test datasets cannot be empty")
        
        for df, name in [(train_data, 'train'), (test_data, 'test')]:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in {name} data")
        
        self._features = self._validator.validate_features(
            features, 
            pd.concat([train_data, test_data]), 
            target_column, 
            prob_cols
        )
        
        self._train_data = train_data.copy()
        self._test_data = test_data.copy()
        self._original_data = pd.concat([train_data, test_data], ignore_index=True)
        self._data = self._original_data[self._features].copy()

    def _validate_categorical_features(self, categorical_features: t.List[str]) -> t.List[str]:
        """Validate provided categorical features."""
        invalid_features = set(categorical_features) - set(self._features)
        if invalid_features:
            raise ValueError(f"Categorical features {invalid_features} not found in features list")
        return categorical_features
    
    def _generate_synthetic_data(self) -> None:
        """
        Generate synthetic data based on the original data distribution.
        Only generates data for the specified features and target column.
        Uses the SyntheticDataGenerator for creating high-quality synthetic data.
        This method gets called when synthetic=True in the initialization.
        """
        try:
            # Get only the relevant columns (features + target)
            relevant_columns = self._features + [self._target_column]
            data_for_synthetic = self._original_data[relevant_columns].copy()
            
            # Create and fit the generator
            generator = SyntheticDataGenerator(preserve_correlations=True)
            generator.fit(
                data=data_for_synthetic,
                target_column=self._target_column,
                model=self._model_handler.model
            )
            
            # Generate synthetic data with a fixed random seed for reproducibility
            random_seed = 42 if not hasattr(self, '_random_state') or self._random_state is None else self._random_state
            self._synthetic_data = generator.generate(
                num_samples=self._synthetic_sample, 
                random_state=random_seed
            )
            
            # Add attribute to indicate synthetic data was generated
            self._has_synthetic_data = True
            
            # Calculate probabilities for synthetic data using the loaded model
            if self._model_handler.model is not None:
                try:
                    # Generate predictions for synthetic data
                    synthetic_proba = self._model_handler.model.predict_proba(self._synthetic_data[self._features])
                    
                    # Create DataFrame with probability columns
                    prob_cols = [f'prob_class_{i}' for i in range(synthetic_proba.shape[1])]
                    synthetic_predictions = pd.DataFrame(
                        synthetic_proba,
                        columns=prob_cols
                    )
                    
                    # Store synthetic predictions
                    self._synthetic_predictions = synthetic_predictions
                    
                    # Update original_prob to use synthetic predictions when synthetic=True
                    if hasattr(self._model_handler, '_predictions') and self._model_handler._predictions is not None:
                        # Store the original predictions temporarily
                        self._model_handler._original_predictions = self._model_handler._predictions
                    
                    # Set the predictions in model_handler to the synthetic ones
                    self._model_handler._predictions = synthetic_predictions
                    self._model_handler._prob_cols = prob_cols
                    
                    print(f"Generated probability predictions for {len(self._synthetic_data)} synthetic samples")
                except Exception as e:
                    print(f"Error calculating probabilities for synthetic data: {str(e)}")
            
            # Evaluate synthetic data quality
            try:
                quality_metrics = generator.evaluate_quality(
                    real_data=data_for_synthetic,
                    synthetic_data=self._synthetic_data
                )
                
                # Store quality metrics
                self._synthetic_quality_metrics = quality_metrics
                
                # Print basic generation information
                print(f"Generated {len(self._synthetic_data)} synthetic samples for {len(self._features)} features and target column '{self._target_column}'")
                
                # Print quality evaluation results
                print("\nSynthetic Data Quality Evaluation:")
                
                # Print overall metrics if available
                if 'overall' in quality_metrics:
                    overall = quality_metrics['overall']
                    print(f"Overall metrics:")
                    for metric, value in overall.items():
                        print(f"  - {metric}: {value:.4f}")
                
                # Print metrics for numerical features
                numerical_metrics = {k: v for k, v in quality_metrics.items() 
                                    if k in self._feature_manager.numerical_features or k == self._target_column}
                if numerical_metrics:
                    print("\nNumerical features:")
                    for feature, metrics in numerical_metrics.items():
                        print(f"  {feature}:")
                        print(f"    - mean diff: {metrics.get('mean_diff', 'N/A'):.4f}")
                        print(f"    - std diff: {metrics.get('std_diff', 'N/A'):.4f}")
                        if 'ks_statistic' in metrics:
                            print(f"    - KS statistic: {metrics['ks_statistic']:.4f}")
                
                # Print metrics for categorical features
                categorical_metrics = {k: v for k, v in quality_metrics.items() 
                                    if k in self._feature_manager.categorical_features}
                if categorical_metrics:
                    print("\nCategorical features:")
                    for feature, metrics in categorical_metrics.items():
                        print(f"  {feature}:")
                        print(f"    - distribution difference: {metrics.get('distribution_difference', 'N/A'):.4f}")
                
            except Exception as e:
                print(f"Error evaluating synthetic data quality: {str(e)}")
            
        except Exception as e:
            print(f"Error generating synthetic data: {str(e)}")
            import traceback
            traceback.print_exc()
            self._has_synthetic_data = False

    @property
    def X(self) -> pd.DataFrame:
        """Return the features dataset (original or synthetic depending on synthetic flag)."""
        if self._synthetic and hasattr(self, '_has_synthetic_data') and self._has_synthetic_data:
            # Return synthetic features if synthetic=True
            return self._synthetic_data[self._features]
        else:
            # Return original features otherwise
            return self._data

    @property
    def target(self) -> pd.Series:
        """Return the target column values (original or synthetic depending on synthetic flag)."""
        if self._synthetic and hasattr(self, '_has_synthetic_data') and self._has_synthetic_data:
            # Return synthetic target if synthetic=True
            return self._synthetic_data[self._target_column]
        else:
            # Return original target otherwise
            return self._original_data[self._target_column]
    
    @property
    def original_prob(self) -> t.Optional[pd.DataFrame]:
        """Return predictions DataFrame (original or synthetic depending on synthetic flag)."""
        if self._synthetic and hasattr(self, '_synthetic_predictions'):
            # Return synthetic predictions if synthetic=True and they exist
            return self._synthetic_predictions
        else:
            # Otherwise, return the predictions from model_handler
            return self._model_handler.predictions

    @property
    def train_data(self) -> pd.DataFrame:
        """Return the training dataset."""
        return self._train_data

    @property
    def test_data(self) -> pd.DataFrame:
        """Return the test dataset."""
        return self._test_data
    
    @property
    def synthetic_data(self) -> t.Optional[pd.DataFrame]:
        """Return the synthetic dataset if available."""
        if hasattr(self, '_has_synthetic_data') and self._has_synthetic_data:
            return self._synthetic_data
        return None
    
    @property
    def synthetic_quality_metrics(self) -> dict:
        """Return quality metrics for synthetic data if available."""
        if hasattr(self, '_synthetic_quality_metrics'):
            return self._synthetic_quality_metrics
        return None

    @property
    def features(self) -> t.List[str]:
        """Return list of feature names."""
        return self._feature_manager.features

    @property
    def categorical_features(self) -> t.List[str]:
        """Return list of categorical feature names."""
        return self._feature_manager.categorical_features

    @property
    def numerical_features(self) -> t.List[str]:
        """Return list of numerical feature names."""
        return self._feature_manager.numerical_features

    @property
    def target_name(self) -> str:
        """Return name of target column."""
        return self._target_column

    @property
    def model(self) -> t.Any:
        """Return the loaded model if available."""
        return self._model_handler.model
    
    @property
    def synthetic_enabled(self) -> bool:
        """Return whether synthetic data generation is enabled."""
        return self._synthetic
    
    @property
    def synthetic_sample_size(self) -> int:
        """Return the size of synthetic sample."""
        return self._synthetic_sample

    def get_feature_data(self, dataset: str = 'train') -> pd.DataFrame:
        """Get feature columns from specified dataset."""
        if dataset.lower() not in ['train', 'test', 'synthetic']:
            raise ValueError("dataset must be either 'train', 'test', or 'synthetic'")
        
        if dataset.lower() == 'synthetic':
            if not hasattr(self, '_has_synthetic_data') or not self._has_synthetic_data:
                raise ValueError("Synthetic data is not available. Initialize with synthetic=True")
            return self._synthetic_data[self._features]
        
        data = self._train_data if dataset.lower() == 'train' else self._test_data
        return data[self._features]

    def get_target_data(self, dataset: str = 'train') -> pd.Series:
        """Get target column from specified dataset."""
        if dataset.lower() not in ['train', 'test', 'synthetic']:
            raise ValueError("dataset must be either 'train', 'test', or 'synthetic'")
        
        if dataset.lower() == 'synthetic':
            if not hasattr(self, '_has_synthetic_data') or not self._has_synthetic_data:
                raise ValueError("Synthetic data is not available. Initialize with synthetic=True")
            return self._synthetic_data[self._target_column]
        
        data = self._train_data if dataset.lower() == 'train' else self._test_data
        return data[self._target_column]

    def set_model(self, model_or_path: t.Union[str, Path, t.Any]) -> None:
        """
        Load and set a model from file or directly set a model object.
        
        Args:
            model_or_path: Either a path to a model file or a model object
        """
        if isinstance(model_or_path, (str, Path)):
            # Load model from path
            self._model_handler.load_model(
                model_or_path,
                features=self._features,
                data={'train': self._train_data, 'test': self._test_data}
            )
        else:
            # Set model directly
            self._model_handler.model = model_or_path
            
            # Generate predictions if possible
            if self._train_data is not None or self._test_data is not None:
                try:
                    data_dict = {}
                    if self._train_data is not None:
                        data_dict['train'] = self._train_data
                    if self._test_data is not None:
                        data_dict['test'] = self._test_data
                        
                    self._model_handler.generate_predictions(data_dict, self._features)
                except Exception as e:
                    print(f"Warning: Could not generate predictions for the new model: {str(e)}")

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self._data)
    
    def __repr__(self) -> str:
        """Return string representation of the dataset."""
        base_repr = self._formatter.format_dataset_info(
            data=self._data if hasattr(self, '_data') else None,
            train_data=self._train_data,
            test_data=self._test_data
        )
        
        # Add synthetic data information if applicable
        if self._synthetic:
            synthetic_info = f"\nSynthetic: Enabled (sample size: {self._synthetic_sample})"
            if hasattr(self, '_has_synthetic_data') and self._has_synthetic_data:
                synthetic_info += f", {len(self._synthetic_data)} samples generated"
            else:
                synthetic_info += ", generation failed"
            
            return base_repr + synthetic_info
        else:
            return base_repr