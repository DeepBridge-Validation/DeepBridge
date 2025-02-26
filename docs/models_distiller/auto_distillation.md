# Minimization Process in AutoDistiller

Looking specifically at the `AutoDistiller` class, the minimization process is orchestrated as follows:

## The AutoDistiller Minimization Pipeline

The `AutoDistiller` class acts as a high-level orchestrator that automates the entire knowledge distillation process, systematically exploring different configurations to find the optimal student model. The actual minimization happens through several coordinated steps:

### 1. Configuration of Search Space

First, the `AutoDistiller` defines the search space of parameters to explore:

```python
def _set_default_config(self):
    """Define default configurations for model types, temperatures, and alphas."""
    self.model_types = [
        ModelType.LOGISTIC_REGRESSION,
        ModelType.DECISION_TREE,
        ModelType.GBM,
        ModelType.XGB
    ]
    
    self.temperatures = [0.5, 1.0, 2.0, 3.0]
    self.alphas = [0.3, 0.5, 0.7, 0.9]
```

This creates a grid of configurations to explore (model types × temperatures × alpha values).

### 2. Experiment Execution 

The `run` method in `AutoDistiller` delegates the experiment execution to `ExperimentRunner.run_experiments()`:

```python
def run(self, use_probabilities: bool = True, verbose_output: bool = False) -> pd.DataFrame:
    # Run experiments
    self.results_df = self.experiment_runner.run_experiments(
        use_probabilities=use_probabilities
    )
```

### 3. Systematic Grid Search

Within `ExperimentRunner.run_experiments()`, a grid search is performed over all configurations:

```python
def run_experiments(self, use_probabilities: bool = True) -> pd.DataFrame:
    # Test all combinations
    for model_type in self.config.model_types:
        for temperature in self.config.temperatures:
            for alpha in self.config.alphas:
                self.config.log_info(f"Testing: {model_type.name}, temp={temperature}, alpha={alpha}")
                
                result = self._run_single_experiment(
                    model_type=model_type,
                    temperature=temperature,
                    alpha=alpha,
                    use_probabilities=use_probabilities
                )
                self.results.append(result)
```

### 4. Individual Experiment Execution

For each configuration, `_run_single_experiment` is called, which initiates the distillation process:

```python
def _run_single_experiment(self, model_type: ModelType, temperature: float, alpha: float, use_probabilities: bool) -> Dict[str, Any]:
    try:
        self.experiment.fit(
            student_model_type=model_type,
            temperature=temperature,
            alpha=alpha,
            use_probabilities=use_probabilities,
            n_trials=self.config.n_trials,
            validation_split=self.config.validation_split,
            verbose=False
        )
        # Get metrics and store results...
    except Exception as e:
        # Handle errors...
```

### 5. Distillation Model Fitting

The actual minimization happens in the `experiment.fit()` method, which delegates to the `KnowledgeDistillation` class:

```python
def fit(self, student_model_type: ModelType, temperature: float, alpha: float, use_probabilities: bool, n_trials: int, validation_split: float, verbose: bool) -> 'Experiment':
    if use_probabilities:
        # Create distillation model from probabilities
        self.distillation_model = KnowledgeDistillation.from_probabilities(
            probabilities=self.prob_train,
            student_model_type=student_model_type,
            student_params=student_params,
            temperature=temperature,
            alpha=alpha,
            n_trials=n_trials,
            validation_split=validation_split,
            random_state=self.random_state
        )
    else:
        # Create distillation model from teacher model
        self.distillation_model = KnowledgeDistillation(
            teacher_model=self.dataset.model,
            student_model_type=student_model_type,
            # Other parameters...
        )
    
    # Train the model
    self.distillation_model.fit(self.X_train, self.y_train, verbose=verbose)
```

### 6. Two-level Optimization

The minimization process occurs at two levels:

1. **Outer loop (grid search)**: Systematically explores different model types, temperatures, and alpha values
2. **Inner loop (hyperparameter optimization)**: For each configuration, Optuna is used to find the optimal hyperparameters for the student model

The inner optimization happens inside `KnowledgeDistillation.fit()`:

```python
def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> 'KnowledgeDistillation':
    # Generate soft labels
    soft_labels = self._get_teacher_soft_labels(X)
    
    if self.student_params is None:
        # Optimize hyperparameters using Optuna
        study = optuna.create_study(direction="minimize")
        objective = lambda trial: self._objective(trial, X, y, soft_labels)
        study.optimize(objective, n_trials=self.n_trials)
        
        # Get the best hyperparameters
        self.best_params = study.best_params
```

### 7. Finding the Best Model

After all experiments are complete, the `AutoDistiller` can identify the best configuration using specific metrics:

```python
def find_best_model(self, metric: str = 'test_accuracy', minimize: bool = False) -> Dict:
    best_config = self.metrics_evaluator.find_best_model(metric=metric, minimize=minimize)
    # Process and return best configuration...
```

Different metrics can be used to select the best model:
- Performance metrics (accuracy, F1, etc.)
- Distribution similarity metrics (KL divergence, KS statistic, R²)

## Summary of the Minimization Process

The minimization in `AutoDistiller` follows these steps:

1. **Define a search space** across model types, temperatures, and alpha values
2. **Systematically explore** all combinations in this search space
3. For each combination:
   - Generate soft labels using the specified temperature
   - Use Optuna to find optimal hyperparameters that minimize the combined loss function
   - Train the student model with the best hyperparameters
   - Evaluate performance and distribution similarity metrics
4. **Select the best configuration** based on the desired metrics

This dual-level optimization (grid search + hyperparameter optimization) efficiently finds an optimal student model that balances performance and model complexity while effectively capturing the teacher's knowledge.