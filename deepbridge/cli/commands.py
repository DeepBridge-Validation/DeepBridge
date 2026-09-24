#!/usr/bin/env python3
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from deepbridge.core.experiment import Experiment

# Initialize Typer app and Rich console
app = typer.Typer(help='DeepBridge CLI - Tools for Model Validation')
validation_app = typer.Typer(help='Model validation commands')
app.add_typer(validation_app, name='validation')
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def setup_experiment(name: str, path: Optional[Path]) -> Experiment:
    """Helper function to create and setup an experiment"""
    try:
        experiment = Experiment(experiment_name=name, save_path=path)
        logger.info(f'Created experiment: {name}')
        return experiment
    except Exception as e:
        logger.error(f'Failed to create experiment: {str(e)}')
        raise typer.Exit(code=1)


# Validation Commands
@validation_app.command('create')
def create_experiment(
    name: str = typer.Argument(..., help='Name of the experiment'),
    path: Optional[Path] = typer.Option(
        None,
        '--path',
        '-p',
        help='Path to save experiment files',
        dir_okay=True,
        file_okay=False,
    ),
):
    """Create a new validation experiment"""
    try:
        experiment = setup_experiment(name, path)
        console.print(
            f"[green]✓[/green] Created experiment '{name}' at {experiment.save_path}"
        )
    except Exception as e:
        console.print(f'[red]✗[/red] Error: {str(e)}')
        raise typer.Exit(code=1)


@validation_app.command('add-data')
def add_data(
    experiment_path: Path = typer.Argument(
        ..., help='Path to experiment directory'
    ),
    train_data: Path = typer.Argument(..., help='Path to training data CSV'),
    target_column: str = typer.Option(
        ..., '--target', '-y', help='Name of target column'
    ),
    test_data: Optional[Path] = typer.Option(
        None, '--test', '-t', help='Path to test data CSV'
    ),
):
    """Add data to an existing experiment"""
    try:
        # Load experiment
        experiment = Experiment(save_path=experiment_path)

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
        console.print('[green]✓[/green] Successfully added data to experiment')

    except Exception as e:
        console.print(f'[red]✗[/red] Error: {str(e)}')
        raise typer.Exit(code=1)


@validation_app.command('info')
def experiment_info(
    experiment_path: Path = typer.Argument(
        ..., help='Path to experiment directory'
    ),
    output_format: str = typer.Option(
        'table', '--format', '-f', help='Output format (table or json)'
    ),
):
    """Get information about an experiment"""
    try:
        experiment = Experiment(save_path=experiment_path)
        info = experiment.get_experiment_info()

        if output_format == 'json':
            console.print_json(data=info)
        else:
            # Create Rich table
            table = Table(title='Experiment Information')
            table.add_column('Property', style='cyan')
            table.add_column('Value', style='magenta')

            # Add rows
            table.add_row('Experiment Name', info['experiment_name'])
            table.add_row('Save Path', str(info['save_path']))
            table.add_row('Number of Models', str(info['n_models']))
            table.add_row(
                'Number of Surrogate Models', str(info['n_surrogate_models'])
            )

            # Add data shapes
            for name, shape in info['data_shapes'].items():
                if shape:
                    table.add_row(f'Shape of {name}', f'{shape}')

            console.print(table)

    except Exception as e:
        console.print(f'[red]✗[/red] Error: {str(e)}')
        raise typer.Exit(code=1)


def version_callback(value: bool):
    """Callback for --version flag"""
    if value:
        console.print('DeepBridge version 0.1.0')
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        '--version',
        '-v',
        help='Show version and exit',
        callback=version_callback,
        is_eager=True,
    )
):
    """DeepBridge CLI - Tools for Model Validation"""
    pass


if __name__ == '__main__':
    app()
