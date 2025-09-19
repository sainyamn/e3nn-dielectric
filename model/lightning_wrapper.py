import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam, lr_scheduler
from typing import Dict, List, Optional, Tuple
import numpy as np
import mlflow
from .nequip_model import NequIPConvolution
from .becs_eps_model import BECS_EPS_nequip_base_Model

class E3NNLightning(pl.LightningModule):
    def __init__(
        self,
        # Model parameters
        hidden_irreps: str = '48x0e + 48x0o + 32x1o + 32x1e + 24x2o + 24x2e',
        num_species: int = 100,
        r_max: float = 5.0,
        use_sc: bool = True,
        graph_net_steps: int = 3,
        
        # Training parameters
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        
        # Data parameters
        batch_size: int = 32,
        num_workers: int = 4,
        
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize the model
        self.model = BECS_EPS_nequip_base_Model(
            hidden_irreps=hidden_irreps,
            num_species=num_species,
            r_max=r_max,
            use_sc=use_sc,
            graph_net_steps=graph_net_steps,
            **kwargs
        )
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Learning rate scheduler settings
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers

    def forward(self, graph):
        return self.model(graph)
    
    def training_step(self, batch, batch_idx):
        # Unpack batch
        vectors = batch['vectors']
        node_specie = batch['node_specie']
        senders = batch['senders']
        receivers = batch['receivers']
        target = batch['target']
        
        # Forward pass
        pred_becs, pred_eps, pred_denoising = self.model(vectors, node_specie, senders, receivers)
        
        # Calculate losses
        loss_becs = self.loss_fn(pred_becs, target['becs'])
        loss_eps = self.loss_fn(pred_eps, target['eps'])
        loss_denoising = self.loss_fn(pred_denoising, target['denoising'])
        
        # Total loss (you can weight these as needed)
        loss = loss_becs + loss_eps + loss_denoising
        
        # Log metrics to MLflow
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_becs', loss_becs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_eps', loss_eps, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_denoising', loss_denoising, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Log to MLflow
        if self.global_step % 10 == 0:  # Log every 10 steps to avoid too much logging
            mlflow.log_metrics({
                'train/loss': loss.item(),
                'train/loss_becs': loss_becs.item(),
                'train/loss_eps': loss_eps.item(),
                'train/loss_denoising': loss_denoising.item(),
                'epoch': float(self.current_epoch)
            }, step=self.global_step)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Similar to training_step but with validation metrics
        vectors = batch['vectors']
        node_specie = batch['node_specie']
        senders = batch['senders']
        receivers = batch['receivers']
        target = batch['target']
        
        with torch.no_grad():
            pred_becs, pred_eps, pred_denoising = self.model(vectors, node_specie, senders, receivers)
            
            # Calculate losses
            loss_becs = self.loss_fn(pred_becs, target['becs'])
            loss_eps = self.loss_fn(pred_eps, target['eps'])
            loss_denoising = self.loss_fn(pred_denoising, target['denoising'])
            loss = loss_becs + loss_eps + loss_denoising
            
            # Log metrics to MLflow
            self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_loss_becs', loss_becs, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_loss_eps', loss_eps, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_loss_denoising', loss_denoising, on_epoch=True, prog_bar=True, logger=True)
            
            # Log to MLflow
            mlflow.log_metrics({
                'val/loss': loss.item(),
                'val/loss_becs': loss_becs.item(),
                'val/loss_eps': loss_eps.item(),
                'val/loss_denoising': loss_denoising.item(),
                'epoch': float(self.current_epoch)
            }, step=self.global_step)
            
            # You can add more metrics like MAE, RMSE, etc.
            
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Learning rate scheduler with warmup
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return float(epoch) / float(max(1, self.warmup_epochs))
            # Cosine annealing after warmup
            progress = float(epoch - self.warmup_epochs) / float(max(1, self.max_epochs - self.warmup_epochs))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def train_dataloader(self):
        # You'll need to implement this based on your data loading
        # Return a PyTorch DataLoader for training
        pass
    
    def val_dataloader(self):
        # Return a PyTorch DataLoader for validation
        pass

def train_model(
    train_loader,
    val_loader,
    config: dict,
    logger=None,
    callbacks=None,
    **trainer_kwargs
):
    """
    Train the model with PyTorch Lightning
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Dictionary with hyperparameters
        logger: Lightning logger (e.g., TensorBoard)
        callbacks: List of callbacks
        **trainer_kwargs: Additional arguments for the Trainer
    """
    # Initialize model
    model = E3NNLightning(**config)
    
    # Default callbacks
    if callbacks is None:
        from pytorch_lightning.callbacks import (
            ModelCheckpoint,
            EarlyStopping,
            LearningRateMonitor
        )
        
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints/',
            filename='e3nn-{epoch:02d}-{val_loss:.4f}',
            save_top_k=3,
            mode='min',
        )
        
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=20,
            verbose=True,
            mode='min'
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.get('max_epochs', 100),
        gpus=1 if torch.cuda.is_available() else 0,
        logger=logger or True,  # Default to TensorBoard
        callbacks=callbacks,
        log_every_n_steps=10,
        **trainer_kwargs
    )
    
    # Train the model
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    return model
