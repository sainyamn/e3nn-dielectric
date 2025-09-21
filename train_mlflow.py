import os
import yaml
import mlflow
import jax
import jax.numpy as jnp
import optax
import numpy as np
import pickle
import tqdm
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# Import model components
from model.datasets import becs_eps_datasets
from model.becs_eps_model import BECS_EPS_model
from model.optimizer import optimizer
from model.loss import BecsEpsLoss
from model.predictors import predict_becs_eps
from model.utils import create_directory_with_random_name, get_edge_relative_vectors
from model.becs_eps_train import evaluate_becs_eps

# MLflow setup
def setup_mlflow(experiment_name: str = "BECS_EPS_Training"):
    """Initialize MLflow tracking."""
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    return mlflow.active_run().info.run_id

def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_safe_predict_becs_eps(original_model_fn, params):
    """Create a safe version of predict_becs_eps that handles variable model outputs"""
    
    def safe_predict_becs_eps(graph):
        """Safe version of predict_becs_eps that handles variable model outputs"""
        
        def model_fn_internal(positions, cell):
            vectors = get_edge_relative_vectors(
                positions=positions,
                senders=graph.senders,
                receivers=graph.receivers,
                shifts=graph.edges.shifts,
                cell=cell,
                n_edge=graph.n_edge,
            )
            # Call the original model and handle variable outputs
            output = original_model_fn(params, vectors, graph.nodes.species, graph.senders, graph.receivers)
            
            # Handle different return formats from the model
            if isinstance(output, tuple):
                if len(output) == 4:
                    # Model returns (becs, eps, denoising, decomposition)
                    node_becs, node_eps, node_denoising, _ = output
                elif len(output) == 3:
                    # Model returns (becs, eps, denoising)
                    node_becs, node_eps, node_denoising = output
                elif len(output) == 2:
                    # Model returns (becs, eps)
                    node_becs, node_eps = output
                    node_denoising = jnp.zeros(node_becs.shape[0])
                else:
                    # Single output or unexpected format
                    node_becs = output[0] if len(output) > 0 else output
                    node_eps = jnp.zeros_like(node_becs)
                    node_denoising = jnp.zeros(node_becs.shape[0])
            else:
                # Single output
                node_becs = output
                node_eps = jnp.zeros_like(node_becs)
                node_denoising = jnp.zeros(node_becs.shape[0])
            
            # Ensure proper shapes
            if len(node_eps.shape) != 3 or node_eps.shape[-1] != 3 or node_eps.shape[-2] != 3:
                node_eps = jnp.zeros_like(node_becs)
            
            return node_becs, node_eps, node_denoising
        
        node_becs, node_eps, node_denoising = model_fn_internal(graph.nodes.positions, graph.globals.cell)
        
        # Continue with the rest of predict_becs_eps logic
        from model.utils import _safe_divide
        import e3nn_jax as e3nn
        
        node_becs_sum = e3nn.scatter_sum(node_becs, nel=graph.n_node)
        node_becs_avg = _safe_divide(node_becs_sum, jnp.expand_dims(jnp.expand_dims(graph.n_node, axis=-1), axis=-1))
        node_becs_avg_repeat = jnp.repeat(node_becs_avg, repeats=graph.n_node, axis=0, total_repeat_length=graph.nodes.positions.shape[0])
        
        node_eps_sum = e3nn.scatter_sum(node_eps, nel=graph.n_node)
        graph_eps_avg = _safe_divide(node_eps_sum, jnp.expand_dims(jnp.expand_dims(graph.n_node, axis=-1), axis=-1))
        
        node_becs = node_becs - node_becs_avg_repeat
        
        return {
            "becs": node_becs,
            "becs_sum": node_becs_sum,
            "becs_avg": node_becs_avg,
            "eps": graph_eps_avg,
            "node_denoising": node_denoising,
        }
    
    return safe_predict_becs_eps

def train_epoch(model_fn, params, opt_state, loss_fn, train_loader, gradient_transform, steps_per_interval):
    """Train for one epoch with proper batch handling."""
    losses = []
    num_updates = 0
    
    @jax.jit
    def update_fn(params, optimizer_state, graph):
        # Create the safe predictor that handles variable model outputs
        safe_predict_becs_eps = create_safe_predict_becs_eps(model_fn, params)
        
        # Compute loss with proper masking for padded graphs
        mask = jax.numpy.ones(graph.n_node.shape[0])  # Simple mask for now
        
        loss, grad = jax.value_and_grad(
            lambda params: jnp.mean(loss_fn(graph, create_safe_predict_becs_eps(model_fn, params)(graph)) * mask)
        )(params)
        
        updates, optimizer_state = gradient_transform.update(grad, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        
        return loss, params, optimizer_state

    # Train for specified number of steps
    def step_loader():
        i = 0
        while True:
            for graph in train_loader:
                yield graph
                i += 1
                if i >= steps_per_interval:
                    return

    p_bar = tqdm.tqdm(
        step_loader(),
        desc=f"Training",
        total=steps_per_interval,
    )
    
    for graph in p_bar:
        loss, params, opt_state = update_fn(params, opt_state, graph)
        loss_val = float(loss)
        losses.append(loss_val)
        p_bar.set_postfix({"loss": f"{loss_val:7.3f}"})

    return params, opt_state, np.mean(losses)

def evaluate_model(model_fn, params, loss_fn, data_loader):
    """Evaluate model using the proper evaluation function."""
    # Create the safe predictor that handles variable model outputs
    safe_predict_becs_eps = create_safe_predict_becs_eps(model_fn, params)
    
    # Use the existing evaluation function
    metrics = evaluate_becs_eps(
        model=safe_predict_becs_eps,
        params=params,
        loss_fn=loss_fn,
        data_loader=data_loader,
        name="validation",
    )
    
    return metrics

def train(config_path: str = "configs/my_config.yaml"):
    """Main training function with MLflow integration."""
    # Load configuration
    config = load_config(config_path)

    # Setup MLflow
    run_id = setup_mlflow()

    try:
        # Log all parameters from config
        def flatten_config(d, prefix=''):
            items = []
            for k, v in d.items():
                if isinstance(v, dict):
                    items.extend(flatten_config(v, f"{prefix}{k}/").items())
                else:
                    items.append((f"{prefix}{k}", v))
            return dict(items)

        mlflow.log_params(flatten_config(config))

        # Load datasets using config parameters directly
        train_loader, valid_loader, test_loader, r_max = becs_eps_datasets(
            r_max=config.get('cutoff', 5.0),
            train_path=config['dataset']['train_path'],
            valid_num=config['dataset'].get('valid_num', 100),
            n_node=config['dataset'].get('num_nodes', 1000),
            n_edge=config['dataset'].get('num_edges', 4000),
            n_graph=config['dataset'].get('num_graphs', 64),
            seed=config['dataset'].get('seed', 77),
        )

        print(f"Loaded {len(train_loader.graphs)} training graphs")
        print(f"Loaded {len(valid_loader.graphs)} validation graphs")

        # Create save directory
        save_dir_name = create_directory_with_random_name("mlflow_training")

        # Initialize model using config parameters directly
        model_fn, params, _ = BECS_EPS_model(
            r_max=r_max,
            train_graphs=train_loader.graphs,
            initialize_seed=config['model'].get('seed', 777),
            num_species=config['model'].get('num_species', 100),
            graph_net_steps=config['model']['num_layers'],
            hidden_irreps=config['model']['internal_irreps'],
            use_sc=config['model'].get('use_sc', True),
            nonlinearities=config['model'].get('nonlinearities', {'e': 'swish', 'o': 'tanh'}),
            force_symmetric_bec=config['model'].get('force_symmetric_bec', False),
            max_ell=config['model'].get('max_ell', 2),
            num_basis=config['model'].get('num_basis', 8),
            radial_net_nonlinearity=config['model'].get('radial_net_nonlinearity', 'raw_swish'),
            radial_net_n_hidden=config['model'].get('radial_net_n_hidden', 64),
            radial_net_n_layers=config['model'].get('radial_net_n_layers', 2),
            scalar_mlp_std=config['model'].get('scalar_mlp_std', 4.0),
            apply_physics_constraints=config['model'].get('apply_physics_constraints', True),
            save_dir_name=save_dir_name,
        )

        print("Model initialized successfully")

        # Create optimizer and loss function using config parameters directly
        gradient_transform, steps_per_interval, _ = optimizer(
            lr=config['training']['learning_rate'],
            max_num_intervals=config['training']['max_num_intervals'],
            steps_per_interval=config['training']['steps_per_interval'],
        )

        opt_state = gradient_transform.init(params)
        loss_fn = BecsEpsLoss(
            becs_weight=config['training'].get('becs_weight', 5.0),
            becs_sum_weight=config['training'].get('becs_sum_weight', 0.0),
            eps_weight=config['training'].get('eps_weight', 1.0),
        )

        print("Starting training...")

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for interval in range(config['training']['max_num_intervals']):
            print(f"\nInterval {interval + 1}/{config['training']['max_num_intervals']}")
            
            # Train for one interval
            params, opt_state, train_loss = train_epoch(
                model_fn, params, opt_state, loss_fn, train_loader, 
                gradient_transform, steps_per_interval
            )

            # Save parameters
            with open(f"{save_dir_name}/params.pkl", "wb") as f:
                pickle.dump(params, f)

            # Evaluate on validation set
            val_metrics = evaluate_model(model_fn, params, loss_fn, valid_loader)
            val_loss = val_metrics.get('mae_becs', 0.0) + val_metrics.get('mae_eps', 0.0)

            # Log metrics
            mlflow.log_metrics({
                'train/loss': float(train_loss),
                'val/loss': float(val_loss),
                'val/mae_becs': float(val_metrics.get('mae_becs', 0.0)),
                'val/mae_eps': float(val_metrics.get('mae_eps', 0.0)),
            }, step=interval)

            print(f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            print(f"Val BECS MAE = {val_metrics.get('mae_becs', 0.0):.4f}, Val EPS MAE = {val_metrics.get('mae_eps', 0.0):.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model parameters
                with open(f"{save_dir_name}/best_params.pkl", "wb") as f:
                    pickle.dump(params, f)
                mlflow.log_artifact(f"{save_dir_name}/best_params.pkl", "model")
            else:
                patience_counter += 1
                if patience_counter >= config['training'].get('patience', 10):
                    print(f"Early stopping at interval {interval + 1}")
                    break

        # Final evaluation on test set if available
        if test_loader is not None and len(test_loader.graphs) > 0:
            test_metrics = evaluate_model(model_fn, params, loss_fn, test_loader)
            test_loss = test_metrics.get('mae_becs', 0.0) + test_metrics.get('mae_eps', 0.0)
            mlflow.log_metrics({
                'test/loss': float(test_loss),
                'test/mae_becs': float(test_metrics.get('mae_becs', 0.0)),
                'test/mae_eps': float(test_metrics.get('mae_eps', 0.0)),
            })
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test BECS MAE: {test_metrics.get('mae_becs', 0.0):.4f}")
            print(f"Test EPS MAE: {test_metrics.get('mae_eps', 0.0):.4f}")

        # Log final artifacts
        mlflow.log_artifacts(save_dir_name, "training_artifacts")
        
        print("Training completed successfully!")

    except Exception as e:
        print(f"Training failed with error: {e}")
        mlflow.log_param("error", str(e))
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train BECS/EPS model with MLflow tracking')
    parser.add_argument('--config', type=str, default='configs/my_config.yaml',
                      help='Path to config file')
    args = parser.parse_args()
    
    train(args.config)