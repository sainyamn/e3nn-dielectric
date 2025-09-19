import os
import argparse
import optuna
import mlflow
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    LearningRateMonitor,
    Callback
)
from optuna.integration import PyTorchLightningPruningCallback
from optuna.integration.mlflow import MLflowCallback
from model.lightning_wrapper import E3NNLightning, train_model
from torch.utils.data import DataLoader, random_split
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict, Any

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def objective(trial, train_loader, val_loader):
    """Objective function for Optuna optimization."""
    # Define hyperparameters to optimize
    config = {
        'hidden_irreps': trial.suggest_categorical('hidden_irreps', [
            '32x0e + 32x0o + 16x1o + 16x1e + 8x2o + 8x2e',
            '48x0e + 48x0o + 24x1o + 24x1e + 16x2o + 16x2e',
            '64x0e + 64x0o + 32x1o + 32x1e + 24x2o + 24x2e'
        ]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'warmup_epochs': trial.suggest_int('warmup_epochs', 5, 20),
        'max_epochs': 100,
        'use_sc': True,
        'graph_net_steps': trial.suggest_int('graph_net_steps', 2, 5),
        'num_species': 100,  # Adjust based on your dataset
        'r_max': 5.0,  # Adjust based on your dataset
    }

    # Set up MLflow experiment
    experiment_name = f'e3nn-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f'trial_{trial.number}'):
        # Log hyperparameters
        mlflow.log_params({
            'hidden_irreps': config['hidden_irreps'],
            'learning_rate': config['learning_rate'],
            'weight_decay': config['weight_decay'],
            'batch_size': config['batch_size'],
            'warmup_epochs': config['warmup_epochs'],
            'graph_net_steps': config['graph_net_steps'],
            'r_max': config['r_max']
        })
        
        # Initialize callbacks list with proper typing
        callbacks: List[Callback] = []
        
        # Add pruning callback if using Optuna
        if trial is not None:
            pruning_callback = PyTorchLightningPruningCallback(
                trial, 
                monitor='val_loss'
            )
            callbacks.append(pruning_callback)
        
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = f'checkpoints/trial_{trial.number if trial is not None else 0}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Add other callbacks
        callbacks.extend([
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                verbose=True,
                mode='min'
            ),
            ModelCheckpoint(
                monitor='val_loss',
                dirpath=checkpoint_dir,
                filename='{epoch:02d}-{val_loss:.4f}',
                save_top_k=1,
                mode='min',
                save_weights_only=True
            ),
            LearningRateMonitor(
                logging_interval='epoch'
            )
        ])

        # Set up device configuration
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
        devices = 1 if torch.cuda.is_available() else 'auto'
        
        # Train the model
        trainer = pl.Trainer(
            max_epochs=config.get('max_epochs', 100),
            accelerator=accelerator,
            devices=devices,
            logger=False,  # Disable default logger since we're using MLflow
            callbacks=callbacks,
            log_every_n_steps=10,
        )
        
        model = train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            trainer=trainer
        )
        
        # Log model architecture
        # Note: You might need to adjust this based on your model's input shape
        # mlflow.pytorch.log_model(model, 'model')
        
        # Log final metrics
        mlflow.log_metrics({
            'best_val_loss': model.best_val_loss if hasattr(model, 'best_val_loss') else float('inf'),
            'epochs_trained': trainer.current_epoch
        })

    # Return the best validation loss
    return model.best_val_loss

def main():
    # Set up MLflow tracking
    mlflow.set_tracking_uri('file:./mlruns')  # Store locally
    
    parser = argparse.ArgumentParser(description='Train e3nn model with PyTorch Lightning and MLflow')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--num_trials', type=int, default=20, help='Number of hyperparameter trials')
    parser.add_argument('--experiment_name', type=str, default='e3nn-experiment', 
                       help='Name for the MLflow experiment')
    args = parser.parse_args()

    # TODO: Load your dataset here
    # train_dataset = YourDataset(os.path.join(args.data_dir, 'train'))
    # val_dataset = YourDataset(os.path.join(args.data_dir, 'val'))
    
    # For demonstration, we'll create dummy data loaders
    # Replace this with your actual data loading code
    train_loader = DataLoader(
        torch.utils.data.TensorDataset(
            torch.randn(1000, 10),  # Replace with your data
            torch.randn(1000, 1)    # Replace with your targets
        ),
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        torch.utils.data.TensorDataset(
            torch.randn(200, 10),   # Replace with your data
            torch.randn(200, 1)     # Replace with your targets
        ),
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    # Set up MLflow experiment
    mlflow.set_experiment(args.experiment_name)
    
    # Set up Optuna study with MLflow callback
    study = optuna.create_study(
        study_name=args.experiment_name,
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
    )
    
    # Add MLflow callback to Optuna
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name='val_loss',
        create_experiment=False
    )

    # Run optimization with MLflow callback
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader),
        n_trials=args.num_trials,
        n_jobs=1,  # Set to >1 for parallel trials if you have multiple GPUs
        callbacks=[mlflow_callback]
    )

    # Print results
    print('\nBest trial:')
    trial = study.best_trial
    print(f'  Value: {trial.value:.4f}')
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    # Save study results
    os.makedirs('studies', exist_ok=True)
    trials_df = study.trials_dataframe()
    trials_df.to_csv('studies/optimization_history.csv')
    
    # Log study results to MLflow
    with mlflow.start_run(run_name='study_summary'):
        # Log best parameters
        mlflow.log_params(study.best_params)
        mlflow.log_metric('best_val_loss', study.best_value)
        
        # Log optimization history
        mlflow.log_artifact('studies/optimization_history.csv')
        
        # Log importance plot
        try:
            import matplotlib.pyplot as plt
            from optuna.visualization import plot_param_importances
            
            fig = plot_param_importances(study)
            plt.savefig('param_importances.png')
            mlflow.log_artifact('param_importances.png')
            os.remove('param_importances.png')
            
            # Log optimization history plot
            from optuna.visualization import plot_optimization_history
            fig = plot_optimization_history(study)
            plt.savefig('optimization_history.png')
            mlflow.log_artifact('optimization_history.png')
            os.remove('optimization_history.png')
            
        except ImportError:
            print("Install plotly and matplotlib for visualization: pip install plotly matplotlib")
    
    print(f"\nMLflow UI can be started with: mlflow ui --backend-store-uri {mlflow.get_tracking_uri()}")

if __name__ == '__main__':
    main()
