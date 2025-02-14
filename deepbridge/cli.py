#!/usr/bin/env python3
import typer
from typing import Optional
from pathlib import Path
import logging
import pandas as pd
import json
from enum import Enum
from rich.console import Console
from rich.table import Table
from deepbridge.model_validation import ModelValidation
from deepbridge.model_distiller import ModelDistiller, ModelType

# Initialize Typer app and Rich console
app = typer.Typer(help="DeepBridge CLI - Tools for Model Validation and Distillation")
validation_app = typer.Typer(help="Model validation commands")
distill_app = typer.Typer(help="Model distillation commands")
app.add_typer(validation_app, name="validation")
app.add_typer(distill_app, name="distill")
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DistillerModelType(str, Enum):
    """Supported model types for distillation"""
    GBM = "gbm"
    XGB = "xgb"
    MLP = "mlp"

def setup_experiment(name: str, path: Optional[Path]) -> ModelValidation:
    """Helper function to create and setup an experiment"""
    try:
        experiment = ModelValidation(experiment_name=name, save_path=path)
        logger.info(f"Created experiment: {name}")
        return experiment
    except Exception as e:
        logger.error(f"Failed to create experiment: {str(e)}")
        raise typer.Exit(code=1)

# Validation Commands
@validation_app.command("create")
def create_experiment(
    name: str = typer.Argument(..., help="Name of the experiment"),
    path: Optional[Path] = typer.Option(
        None, 
        "--path", 
        "-p", 
        help="Path to save experiment files",
        dir_okay=True,
        file_okay=False
    )
):
    """Create a new validation experiment"""
    try:
        experiment = setup_experiment(name, path)
        console.print(f"[green]✓[/green] Created experiment '{name}' at {experiment.save_path}")
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {str(e)}")
        raise typer.Exit(code=1)

@validation_app.command("add-data")
def add_data(
    experiment_path: Path = typer.Argument(..., help="Path to experiment directory"),
    train_data: Path = typer.Argument(..., help="Path to training data CSV"),
    target_column: str = typer.Option(..., "--target", "-y", help="Name of target column"),
    test_data: Optional[Path] = typer.Option(None, "--test", "-t", help="Path to test data CSV")
):
    """Add data to an existing experiment"""
    try:
        # Load experiment
        experiment = ModelValidation(save_path=experiment_path)
        
        # Load training data
        train_df = pd.read_csv(train_data)
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        
        # Load test data if provided
        X_test = None
        y_test = None
        if test_data:
            test_df = pd.read_csv(test_data)
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
        
        # Add data to experiment
        experiment.add_data(X_train, y_train, X_test, y_test)
        console.print("[green]✓[/green] Successfully added data to experiment")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {str(e)}")
        raise typer.Exit(code=1)

@validation_app.command("info")
def experiment_info(
    experiment_path: Path = typer.Argument(..., help="Path to experiment directory"),
    output_format: str = typer.Option(
        "table", 
        "--format", 
        "-f",
        help="Output format (table or json)"
    )
):
    """Get information about an experiment"""
    try:
        experiment = ModelValidation(save_path=experiment_path)
        info = experiment.get_experiment_info()
        
        if output_format == "json":
            console.print_json(data=info)
        else:
            # Create Rich table
            table = Table(title="Experiment Information")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="magenta")
            
            # Add rows
            table.add_row("Experiment Name", info["experiment_name"])
            table.add_row("Save Path", str(info["save_path"]))
            table.add_row("Number of Models", str(info["n_models"]))
            table.add_row("Number of Surrogate Models", str(info["n_surrogate_models"]))
            
            # Add data shapes
            for name, shape in info["data_shapes"].items():
                if shape:
                    table.add_row(f"Shape of {name}", f"{shape}")
            
            console.print(table)
            
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {str(e)}")
        raise typer.Exit(code=1)

# Distillation Commands
@distill_app.command("train")
def train_distilled_model(
    model_type: DistillerModelType = typer.Argument(..., help="Type of model to use"),
    original_predictions: Path = typer.Argument(..., help="Path to original model predictions CSV"),
    features_data: Path = typer.Argument(..., help="Path to features data CSV"),
    save_path: Optional[Path] = typer.Option(None, "--save", "-s", help="Path to save the model"),
    params_file: Optional[Path] = typer.Option(
        None, 
        "--params", 
        "-p", 
        help="JSON file with model parameters"
    ),
    test_size: float = typer.Option(0.2, "--test-size", "-t", help="Test set size for validation")
):
    """Train a distilled model"""
    try:
        # Load data
        predictions = pd.read_csv(original_predictions)
        features = pd.read_csv(features_data)
        
        # Load model parameters if provided
        model_params = None
        if params_file:
            with open(params_file) as f:
                model_params = json.load(f)
        
        with console.status("[bold green]Training model..."):
            # Create and train distiller
            distiller = ModelDistiller(
                model_type=model_type.value,
                model_params=model_params,
                save_path=save_path
            )
            
            distiller.fit(
                X=features,
                probas=predictions,
                test_size=test_size,
                verbose=True
            )
        
        if save_path:
            console.print(f"[green]✓[/green] Model saved to {save_path}")
            
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {str(e)}")
        raise typer.Exit(code=1)

@distill_app.command("predict")
def predict_with_model(
    model_path: Path = typer.Argument(..., help="Path to saved model"),
    input_data: Path = typer.Argument(..., help="Path to input data CSV"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to save predictions")
):
    """Make predictions with a distilled model"""
    try:
        with console.status("[bold green]Loading model and making predictions..."):
            # Load model and data
            distiller = ModelDistiller.load(model_path)
            features = pd.read_csv(input_data)
            
            # Make predictions
            predictions = distiller.predict(features)
        
        # Save or display predictions
        if output:
            pd.DataFrame(predictions, columns=['probability']).to_csv(output, index=False)
            console.print(f"[green]✓[/green] Predictions saved to {output}")
        else:
            console.print("\nPredictions:")
            console.print(predictions)
            
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {str(e)}")
        raise typer.Exit(code=1)

@distill_app.command("evaluate")
def evaluate_model(
    model_path: Path = typer.Argument(..., help="Path to saved model"),
    true_labels: Path = typer.Argument(..., help="Path to true labels CSV"),
    original_predictions: Path = typer.Argument(..., help="Path to original model predictions CSV"),
    distilled_predictions: Path = typer.Argument(..., help="Path to distilled model predictions CSV"),
    output_format: str = typer.Option(
        "table", 
        "--format", 
        "-f",
        help="Output format (table or json)"
    )
):
    """Evaluate distilled model performance"""
    try:
        # Load data
        y_true = pd.read_csv(true_labels).values.ravel()
        original_probs = pd.read_csv(original_predictions).values.ravel()
        distilled_probs = pd.read_csv(distilled_predictions).values.ravel()
        
        with console.status("[bold green]Calculating metrics..."):
            # Calculate metrics
            metrics = ModelDistiller.calculate_detailed_metrics(
                original_probas=original_probs,
                distilled_probas=distilled_probs,
                y_true=y_true
            )
        
        if output_format == "json":
            console.print_json(data=metrics)
        else:
            # Create Rich table for main metrics
            table = Table(title="Model Performance Comparison")
            table.add_column("Metric", style="cyan")
            table.add_column("Original Model", style="green")
            table.add_column("Distilled Model", style="magenta")
            
            # Add main metrics
            table.add_row(
                "ROC AUC",
                f"{metrics['original_roc_auc']:.4f}",
                f"{metrics['distilled_roc_auc']:.4f}"
            )
            table.add_row(
                "Accuracy",
                f"{metrics['original_accuracy']:.4f}",
                f"{metrics['distilled_accuracy']:.4f}"
            )
            table.add_row(
                "Average Precision",
                f"{metrics['original_avg_precision']:.4f}",
                f"{metrics['distilled_avg_precision']:.4f}"
            )
            
            console.print(table)
            
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {str(e)}")
        raise typer.Exit(code=1)

def version_callback(value: bool):
    """Callback for --version flag"""
    if value:
        console.print("DeepBridge version 0.1.0")
        raise typer.Exit()

@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    )
):
    """DeepBridge CLI - Tools for Model Validation and Distillation"""
    pass

if __name__ == "__main__":
    app()