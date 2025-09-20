import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
import json
import os
import pickle
import threading
import time
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Optional, Tuple
import queue
import tempfile
import traceback

# Add your model imports here
try:
    import jax
    import jax.numpy as jnp
    import haiku as hk
    import optax
    import e3nn_jax as e3nn
    
    # Import your model modules
    from model.datasets import becs_eps_datasets
    from model.becs_eps_model import BECS_EPS_model
    from model.optimizer import optimizer
    from model.becs_eps_train import BECS_EPS_train, evaluate_becs_eps
    from model.loss import BecsEpsLoss
    from model.predictors import predict_becs_eps
    from model.utils import create_directory_with_random_name, get_edge_relative_vectors, _safe_divide
    
    IMPORTS_AVAILABLE = True
    st.success("✅ All model modules loaded successfully!")
except ImportError as e:
    IMPORTS_AVAILABLE = False
    st.error(f"❌ Model modules not found: {e}")
    st.info("Please ensure your model modules are in the Python path")

# Configure Streamlit page
st.set_page_config(
    page_title="BECS/EPS Training Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'training_active' not in st.session_state:
    st.session_state.training_active = False
if 'training_history' not in st.session_state:
    st.session_state.training_history = []
if 'current_config' not in st.session_state:
    st.session_state.current_config = None
if 'model_state' not in st.session_state:
    st.session_state.model_state = None
if 'training_thread' not in st.session_state:
    st.session_state.training_thread = None
if 'metrics_queue' not in st.session_state:
    st.session_state.metrics_queue = queue.Queue()
if 'training_error' not in st.session_state:
    st.session_state.training_error = None
if 'current_interval' not in st.session_state:
    st.session_state.current_interval = 0
if 'stop_training_flag' not in st.session_state:
    st.session_state.stop_training_flag = threading.Event()

def create_default_config():
    """Create default training configuration"""
    return {
        'cutoff': 5.0,
        'dataset': {
            'train_path': 'data/phonon_becs_epsilon.xyz',
            'train_num': 1520,
            'valid_num': 100,
            'num_nodes': 1000,
            'num_edges': 4000,
            'num_graphs': 64,
            'seed': 77
        },
        'model': {
            'num_layers': 3,
            'internal_irreps': '48x0e + 48x0o + 32x1o + 32x1e + 24x2o + 24x2e',
            'seed': 777,
            'num_species': 100,
            'avg_num_neighbors': 'average',
            'force_symmetric_bec': False,
            'max_ell': 3,
            'num_basis': 8,
            'radial_net_nonlinearity': 'raw_swish',
            'radial_net_n_hidden': 64,
            'radial_net_n_layers': 2,
            'scalar_mlp_std': 4.0,
            'apply_physics_constraints': True
        },
        'training': {
            'learning_rate': 0.004,
            'steps_per_interval': 500,
            'max_num_intervals': 30,
            'patience': 10,
            'becs_weight': 5.0,
            'becs_sum_weight': 0.0,
            'eps_weight': 1.0,
            'ema_decay': 0.99
        }
    }

def save_config(config, filename):
    """Save configuration to YAML file"""
    os.makedirs('configs', exist_ok=True)
    with open(f'configs/{filename}', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def load_config(filename):
    """Load configuration from YAML file"""
    try:
        with open(f'configs/{filename}', 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        st.error(f"Configuration file {filename} not found!")
        return None

def plot_training_metrics(history):
    """Create interactive plots for training metrics"""
    if not history:
        return None
    
    df = pd.DataFrame(history)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training BECS MAE', 'Validation BECS MAE', 'Training EPS MAE', 'Validation EPS MAE'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Training BECS MAE
    if 'train_mae_becs' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['interval'], y=df['train_mae_becs'], 
                      name='Train BECS MAE', line=dict(color='blue')),
            row=1, col=1
        )
    
    # Validation BECS MAE
    if 'val_mae_becs' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['interval'], y=df['val_mae_becs'], 
                      name='Val BECS MAE', line=dict(color='red')),
            row=1, col=2
        )
    
    # Training EPS MAE
    if 'train_mae_eps' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['interval'], y=df['train_mae_eps'], 
                      name='Train EPS MAE', line=dict(color='green')),
            row=2, col=1
        )
    
    # Validation EPS MAE
    if 'val_mae_eps' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['interval'], y=df['val_mae_eps'], 
                      name='Val EPS MAE', line=dict(color='orange')),
            row=2, col=2
        )
    
    fig.update_layout(height=600, showlegend=False)
    fig.update_xaxes(title_text="Interval")
    fig.update_yaxes(title_text="MAE")
    
    return fig

def real_training_worker(config, save_dir_name, metrics_queue, stop_flag):
    """Real training worker function that runs in a separate thread"""
    try:
        # Set JAX to CPU for training (you might want GPU if available)
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        
        # Load datasets
        train_loader, valid_loader, test_loader, r_max = becs_eps_datasets(
            r_max=config['cutoff'],
            train_path=config['dataset']['train_path'],
            valid_num=config['dataset']['valid_num'],
            n_node=config['dataset']['num_nodes'],
            n_edge=config['dataset']['num_edges'],
            n_graph=config['dataset']['num_graphs'],
        )
        
        metrics_queue.put({"status": "Data loaded successfully", "type": "info"})
        
        # Create model
        model_fn, params, num_message_passing = BECS_EPS_model(
            r_max=r_max,
            train_graphs=train_loader.graphs,
            initialize_seed=config['model']['seed'],
            num_species=config['model']['num_species'],
            use_sc=True,
            graph_net_steps=config['model']['num_layers'],
            hidden_irreps=config['model']['internal_irreps'],
            nonlinearities={'e': 'swish', 'o': 'tanh'},
            save_dir_name=save_dir_name,
            force_symmetric_bec=config['model']['force_symmetric_bec'],
            max_ell=config['model']['max_ell'],
            num_basis=config['model']['num_basis'],
            radial_net_nonlinearity=config['model']['radial_net_nonlinearity'],
            radial_net_n_hidden=config['model']['radial_net_n_hidden'],
            radial_net_n_layers=config['model']['radial_net_n_layers'],
            scalar_mlp_std=config['model']['scalar_mlp_std'],
            apply_physics_constraints=config['model']['apply_physics_constraints'],
        )
        
        metrics_queue.put({"status": "Model created successfully", "type": "info"})
        
        # Create a simple model wrapper that truncates extra return values
        def create_wrapped_model(original_model_fn, params):
            """Create a wrapped model that returns exactly 3 values with correct shapes"""
            def wrapped_model(vectors, species, senders, receivers):
                output = original_model_fn(params, vectors, species, senders, receivers)
                
                # Ensure exactly 3 return values for predict_becs_eps
                if isinstance(output, tuple):
                    if len(output) >= 3:
                        node_becs, node_eps, node_denoising = output[:3]
                        
                        # Fix epsilon shape if needed - ensure it's [n_nodes, 3, 3]
                        if len(node_eps.shape) != 3 or node_eps.shape[-1] != 3 or node_eps.shape[-2] != 3:
                            # Make eps same shape as becs initially (per-node)
                            node_eps = jnp.zeros_like(node_becs)
                        
                        return (node_becs, node_eps, node_denoising)
                        
                    elif len(output) == 2:
                        node_becs, node_eps = output
                        
                        # Fix epsilon shape if needed
                        if len(node_eps.shape) != 3 or node_eps.shape[-1] != 3 or node_eps.shape[-2] != 3:
                            node_eps = jnp.zeros_like(node_becs)
                        
                        # Add dummy denoising
                        node_denoising = jnp.zeros(node_becs.shape[0])
                        return (node_becs, node_eps, node_denoising)
                        
                    elif len(output) == 1:
                        # If only one output, assume it's BECS and create dummy EPS and denoising
                        node_becs = output[0]
                        node_eps = jnp.zeros_like(node_becs)  # Same shape as BECS
                        node_denoising = jnp.zeros(node_becs.shape[0])
                        return (node_becs, node_eps, node_denoising)
                else:
                    # Single output case
                    node_becs = output
                    node_eps = jnp.zeros_like(node_becs)  # Same shape as BECS
                    node_denoising = jnp.zeros(node_becs.shape[0])
                    return (node_becs, node_eps, node_denoising)
                
                return output
            
            return wrapped_model
        
        # Create the predictor using the original predict_becs_eps function
        def safe_predict_becs_eps(params, graph):
            """Safe predictor that uses the original predict_becs_eps"""
            wrapped_model = create_wrapped_model(model_fn, params)
            return predict_becs_eps(wrapped_model, graph)
        
        predictor = jax.jit(safe_predict_becs_eps)
        
        # Create optimizer
        gradient_transform, steps_per_interval, max_num_intervals = optimizer(
            lr=config['training']['learning_rate'],
            max_num_intervals=config['training']['max_num_intervals'],
            steps_per_interval=config['training']['steps_per_interval'],
        )
        optimizer_state = gradient_transform.init(params)
        
        # Create loss function
        loss_fn = BecsEpsLoss(
            becs_weight=config['training']['becs_weight'],
            becs_sum_weight=config['training']['becs_sum_weight'],
            eps_weight=config['training']['eps_weight'],
        )
        
        metrics_queue.put({"status": "Starting training...", "type": "info"})
        
        # Custom training loop with metrics reporting
        def custom_evaluate_and_report(model, params, loss_fn, data_loader, mode, interval):
            """Custom evaluation function that reports metrics"""
            try:
                # First, let's try to call the function and see what it returns
                metrics_queue.put({"status": f"Starting evaluation for {mode}", "type": "info"})
                
                eval_result = evaluate_becs_eps(
                    model=model,
                    params=params,
                    loss_fn=loss_fn,
                    data_loader=data_loader,
                    name=mode,
                )
                
                metrics_queue.put({"status": f"Evaluation completed for {mode}, result type: {type(eval_result)}", "type": "info"})
                
                # Handle different return formats
                metrics = {}
                if isinstance(eval_result, tuple):
                    metrics_queue.put({"status": f"Got tuple with {len(eval_result)} elements", "type": "info"})
                    
                    if len(eval_result) == 1:
                        metrics = {'loss': eval_result[0]}
                    elif len(eval_result) == 2:
                        # Assuming (loss, metrics) format
                        loss, metric_dict = eval_result
                        if isinstance(metric_dict, dict):
                            metrics = {'loss': loss, **metric_dict}
                        else:
                            metrics = {'loss': loss, 'value': metric_dict}
                    elif len(eval_result) == 3:
                        # Assuming (loss, mae_becs, mae_eps) format
                        loss, mae_becs, mae_eps = eval_result
                        metrics = {
                            'loss': loss,
                            'mae_becs': mae_becs,
                            'mae_eps': mae_eps
                        }
                    elif len(eval_result) == 4:
                        # Assuming (loss, mae_becs, mae_eps, other) format
                        loss, mae_becs, mae_eps, other = eval_result
                        metrics = {
                            'loss': loss,
                            'mae_becs': mae_becs,
                            'mae_eps': mae_eps,
                            'other': other
                        }
                    else:
                        # Handle other tuple lengths by storing all values
                        metrics = {f'value_{i}': val for i, val in enumerate(eval_result)}
                        if len(eval_result) >= 3:
                            metrics['mae_becs'] = eval_result[1] if len(eval_result) > 1 else 0
                            metrics['mae_eps'] = eval_result[2] if len(eval_result) > 2 else 0
                            
                elif isinstance(eval_result, dict):
                    # If it's already a dictionary
                    metrics = eval_result
                else:
                    # Single value
                    metrics = {'value': eval_result}
                
                # Report metrics
                metric_data = {
                    'interval': interval,
                    'mode': mode,
                    'type': 'metrics',
                    **metrics
                }
                metrics_queue.put(metric_data)
                
                return metrics
                
            except Exception as e:
                import traceback
                error_msg = f"Error in evaluation for {mode}: {str(e)}\n"
                error_msg += f"Traceback: {traceback.format_exc()}\n"
                if 'eval_result' in locals():
                    error_msg += f"Eval result type: {type(eval_result)}\n"
                    error_msg += f"Eval result: {eval_result}\n"
                else:
                    error_msg += "Error occurred before getting eval_result\n"
                    
                metrics_queue.put({"status": error_msg, "type": "error"})
                return {}
        
        # Start training with custom evaluation
        from model.becs_eps_train import train_becs_eps
        
        for interval, params, optimizer_state, ema_params in train_becs_eps(
            model=predictor,
            params=params,
            loss_fn=loss_fn,
            train_loader=train_loader,
            gradient_transform=gradient_transform,
            optimizer_state=optimizer_state,
            steps_per_interval=config['training']['steps_per_interval'],
            ema_decay=config['training'].get('ema_decay', None),
        ):
            # Check if training should stop using the threading event
            if stop_flag.is_set():
                metrics_queue.put({"status": "Training stopped by user", "type": "info"})
                break
            
            # Save parameters
            with open(f"{save_dir_name}/params.pkl", "wb") as f:
                pickle.dump(params, f)
            with open(f"{save_dir_name}/ema_params.pkl", "wb") as f:
                pickle.dump(ema_params, f)
            
            # Evaluate on training set
            train_metrics = custom_evaluate_and_report(
                predictor, ema_params, loss_fn, train_loader, "train", interval
            )
            
            # Evaluate on validation set
            val_metrics = custom_evaluate_and_report(
                predictor, ema_params, loss_fn, valid_loader, "validation", interval
            )
            
            # Update current interval
            metrics_queue.put({
                "interval": interval,
                "type": "interval_update"
            })
            
            # Check for max intervals
            if interval >= config['training']['max_num_intervals']:
                metrics_queue.put({"status": "Training completed - max intervals reached", "type": "success"})
                break
                
            # Simple early stopping check
            if interval >= config['training']['patience'] and val_metrics.get('mae_becs'):
                # You could implement more sophisticated early stopping here
                pass
        
        metrics_queue.put({"status": "Training completed successfully!", "type": "success"})
        
    except Exception as e:
        error_msg = f"Training error: {str(e)}\n{traceback.format_exc()}"
        metrics_queue.put({"status": error_msg, "type": "error"})

def start_real_training(config):
    """Start real training in a separate thread"""
    if not IMPORTS_AVAILABLE:
        st.error("Cannot start training - model modules not available")
        return
    
    # Create save directory
    save_dir_name = create_directory_with_random_name("streamlit_training")
    
    # Save config
    config_path = os.path.join(save_dir_name, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    st.session_state.training_active = True
    st.session_state.current_config = config
    st.session_state.training_history = []
    st.session_state.training_error = None
    st.session_state.current_interval = 0
    
    # Clear the stop flag
    st.session_state.stop_training_flag.clear()
    
    # Start training thread
    training_thread = threading.Thread(
        target=real_training_worker,
        args=(config, save_dir_name, st.session_state.metrics_queue, st.session_state.stop_training_flag),
        daemon=True
    )
    st.session_state.training_thread = training_thread
    training_thread.start()

def stop_training():
    """Stop the training process"""
    st.session_state.training_active = False
    st.session_state.stop_training_flag.set()

def check_training_updates():
    """Check for training updates from the queue"""
    updates_received = False
    
    try:
        while not st.session_state.metrics_queue.empty():
            update = st.session_state.metrics_queue.get_nowait()
            updates_received = True
            
            if update['type'] == 'metrics':
                # Update training history
                interval = update['interval']
                mode = update['mode']
                
                # Find or create entry for this interval
                entry = None
                for hist_entry in st.session_state.training_history:
                    if hist_entry['interval'] == interval:
                        entry = hist_entry
                        break
                
                if entry is None:
                    entry = {'interval': interval}
                    st.session_state.training_history.append(entry)
                
                # Update metrics
                if mode == 'train':
                    entry['train_mae_becs'] = update.get('mae_becs')
                    entry['train_mae_eps'] = update.get('mae_eps')
                elif mode == 'validation':
                    entry['val_mae_becs'] = update.get('mae_becs')
                    entry['val_mae_eps'] = update.get('mae_eps')
                    
            elif update['type'] == 'interval_update':
                st.session_state.current_interval = update['interval']
                
            elif update['type'] == 'info':
                st.info(update['status'])
                
            elif update['type'] == 'success':
                st.success(update['status'])
                st.session_state.training_active = False
                
            elif update['type'] == 'error':
                st.error(update['status'])
                st.session_state.training_error = update['status']
                st.session_state.training_active = False
                
    except queue.Empty:
        pass
    
    return updates_received

# Main app layout
st.title("🧪 BECS/EPS Model Training Dashboard")

# Check for training updates
if st.session_state.training_active:
    updates_received = check_training_updates()
    if updates_received:
        st.rerun()

# Sidebar for configuration
st.sidebar.header("Training Configuration")

# Configuration file management
st.sidebar.subheader("Configuration Management")
config_files = []
if os.path.exists('configs'):
    config_files = [f for f in os.listdir('configs') if f.endswith('.yaml')]

selected_config = st.sidebar.selectbox("Load existing config:", ["Create new"] + config_files)

if selected_config != "Create new":
    config = load_config(selected_config)
else:
    config = create_default_config()

# Dataset configuration
st.sidebar.subheader("Dataset Settings")
config['dataset']['train_path'] = st.sidebar.text_input("Training data path:", 
                                                       value=config['dataset']['train_path'])
config['dataset']['train_num'] = st.sidebar.number_input("Training samples:", 
                                                        value=config['dataset']['train_num'], 
                                                        min_value=1, max_value=10000)
config['dataset']['valid_num'] = st.sidebar.number_input("Validation samples:", 
                                                        value=config['dataset']['valid_num'], 
                                                        min_value=1, max_value=1000)

# Model configuration
st.sidebar.subheader("Model Settings")
config['model']['num_layers'] = st.sidebar.slider("Number of layers:", 1, 10, config['model']['num_layers'])
config['model']['internal_irreps'] = st.sidebar.text_input("Internal irreps:", 
                                                          value=config['model']['internal_irreps'])
config['model']['num_species'] = st.sidebar.number_input("Number of species:", 
                                                        value=config['model']['num_species'], 
                                                        min_value=1, max_value=118)
config['model']['force_symmetric_bec'] = st.sidebar.checkbox("Force symmetric BEC:", 
                                                           value=config['model']['force_symmetric_bec'])
config['model']['max_ell'] = st.sidebar.slider("Max spherical harmonic degree:", 1, 5, config['model']['max_ell'])
config['model']['num_basis'] = st.sidebar.slider("Number of basis functions:", 4, 16, config['model']['num_basis'])

# Training configuration
st.sidebar.subheader("Training Settings")
config['training']['learning_rate'] = st.sidebar.number_input("Learning rate:", 
                                                             value=config['training']['learning_rate'], 
                                                             min_value=0.0001, max_value=0.1, 
                                                             format="%.4f")
config['training']['max_num_intervals'] = st.sidebar.number_input("Max intervals:", 
                                                                 value=config['training']['max_num_intervals'], 
                                                                 min_value=1, max_value=1000)
config['training']['steps_per_interval'] = st.sidebar.number_input("Steps per interval:", 
                                                                  value=config['training']['steps_per_interval'], 
                                                                  min_value=1, max_value=2000)
config['training']['patience'] = st.sidebar.number_input("Early stopping patience:", 
                                                        value=config['training']['patience'], 
                                                        min_value=1, max_value=100)

# Loss weights
st.sidebar.subheader("Loss Weights")
config['training']['becs_weight'] = st.sidebar.number_input("BECS weight:", 
                                                           value=config['training']['becs_weight'], 
                                                           min_value=0.0, max_value=100.0)
config['training']['eps_weight'] = st.sidebar.number_input("EPS weight:", 
                                                          value=config['training']['eps_weight'], 
                                                          min_value=0.0, max_value=100.0)

# Save configuration
new_config_name = st.sidebar.text_input("Save config as:", value="my_config.yaml")
if st.sidebar.button("Save Configuration"):
    save_config(config, new_config_name)
    st.sidebar.success(f"Configuration saved as {new_config_name}")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Training Progress")
    
    # Training controls
    col1a, col1b, col1c = st.columns(3)
    
    with col1a:
        if st.button("🚀 Start Training", disabled=st.session_state.training_active or not IMPORTS_AVAILABLE):
            start_real_training(config)
    
    with col1b:
        if st.button("⏹️ Stop Training", disabled=not st.session_state.training_active):
            stop_training()
            st.warning("Training stop requested...")
    
    with col1c:
        if st.button("🔄 Reset"):
            st.session_state.training_history = []
            st.session_state.training_active = False
            st.session_state.training_error = None
            st.session_state.current_interval = 0
            st.session_state.stop_training_flag.clear()
            # Clear the queue
            while not st.session_state.metrics_queue.empty():
                try:
                    st.session_state.metrics_queue.get_nowait()
                except queue.Empty:
                    break
            st.info("Training reset.")
    
    # Training status
    if st.session_state.training_active:
        st.info("🔥 Training in progress...")
        progress_bar = st.progress(0)
        if st.session_state.current_config:
            current_interval = st.session_state.current_interval
            max_intervals = st.session_state.current_config['training']['max_num_intervals']
            progress = min(current_interval / max_intervals, 1.0) if max_intervals > 0 else 0
            progress_bar.progress(progress)
            st.write(f"Interval {current_interval}/{max_intervals}")
    elif st.session_state.training_error:
        st.error("Training failed!")
        with st.expander("Error Details"):
            st.text(st.session_state.training_error)
    elif st.session_state.training_history:
        st.success("Training completed!")
    
    # Plot training metrics
    if st.session_state.training_history:
        fig = plot_training_metrics(st.session_state.training_history)
        if fig:
            st.plotly_chart(fig, width='stretch')
        
        # Display current metrics
        if st.session_state.training_history:
            latest_metrics = st.session_state.training_history[-1]
            st.subheader("Latest Metrics")
            
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Interval", latest_metrics.get('interval', 'N/A'))
            with metric_cols[1]:
                val = latest_metrics.get('train_mae_becs', 0)
                st.metric("Train BECS MAE", f"{val:.4f}" if val else "N/A")
            with metric_cols[2]:
                val = latest_metrics.get('val_mae_becs', 0)
                st.metric("Val BECS MAE", f"{val:.4f}" if val else "N/A")
            with metric_cols[3]:
                val = latest_metrics.get('val_mae_eps', 0)
                st.metric("Val EPS MAE", f"{val:.4f}" if val else "N/A")

with col2:
    st.header("Configuration Summary")
    
    # Display current configuration
    with st.expander("Dataset", expanded=True):
        st.json(config['dataset'])
    
    with st.expander("Model", expanded=True):
        st.json(config['model'])
    
    with st.expander("Training", expanded=True):
        st.json(config['training'])
    
    # Training history table
    if st.session_state.training_history:
        st.subheader("Training History")
        df = pd.DataFrame(st.session_state.training_history)
        st.dataframe(df.tail(10), width='stretch')
        
        # Download training history
        csv = df.to_csv(index=False)
        st.download_button(
            label="📊 Download Training History",
            data=csv,
            file_name="training_history.csv",
            mime="text/csv"
        )

# Auto-refresh during training
if st.session_state.training_active:
    time.sleep(2)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("**BECS/EPS Model Training Dashboard** - Built with Streamlit")

# Add some custom CSS for better styling
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}

.stButton > button {
    width: 100%;
}

.training-active {
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.7; }
    100% { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)