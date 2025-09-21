import functools
import math
import os
from typing import Any, Callable, Dict, Optional, Union, Tuple, List
import numpy as np
import pickle

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np

from e3nn_jax import Irreps, Irrep
from e3nn_jax import IrrepsArray
from e3nn_jax.haiku import Linear

from .utils import (
    create_directory_with_random_name,
    compute_avg_num_neighbors,
    bessel_basis,
    soft_envelope,
    safe_norm,
)

from .data_utils import (
    get_atomic_number_table_from_zs,
    compute_average_E0s,
)

from .blocks import (
    RadialEmbeddingBlock,
    LinearNodeEmbeddingBlock,
)

from .nequip_model import (
    NequIPConvolution,
)

partial = functools.partial
Array = jnp.ndarray


class BECS_EPS_nequip_base_Model(hk.Module):
    def __init__(
        self,
        *,
        graph_net_steps: int,
        use_sc: bool,
        nonlinearities: Union[str, Dict[str, str]],
        hidden_irreps: str,
        max_ell: int = 3,
        num_basis: int = 8,
        r_max: float = 4.,
        num_species: int = None,
        avg_r_min: float = None,
        num_features: int = 7,
        radial_basis: Callable[[jnp.ndarray], jnp.ndarray],
        radial_net_nonlinearity: str = 'raw_swish',
        radial_net_n_hidden: int = 64,
        radial_net_n_layers: int = 2,
        radial_envelope: Callable[[jnp.ndarray], jnp.ndarray],
        shift: float = 0.,
        scale: float = 1.,
        avg_num_neighbors: float = 1.,
        scalar_mlp_std: float = 4.0,
        apply_physics_constraints: bool = True,
        force_symmetric_bec: bool = False,  # NEW: Option to force symmetry
        **kwargs,  # Accept extra kwargs but don't use them
    ):
        """
        Improved BECS/EPS model with physics-informed constraints.
        
        Args:
            graph_net_steps: Number of message passing layers
            use_sc: Use self-connections in message passing
            nonlinearities: Activation functions for even/odd irreps
            hidden_irreps: Hidden layer irrep specification
            max_ell: Maximum spherical harmonic degree
            num_basis: Number of radial basis functions
            r_max: Cutoff radius
            num_species: Number of atomic species
            avg_r_min: Average minimum distance (for normalization)
            radial_basis: Radial basis function
            radial_net_nonlinearity: Radial network activation
            radial_net_n_hidden: Radial network hidden size
            radial_net_n_layers: Radial network depth
            radial_envelope: Radial envelope function
            avg_num_neighbors: Average coordination number
            scalar_mlp_std: MLP initialization scale
            apply_physics_constraints: Whether to apply BEC constraints
            force_symmetric_bec: Whether to force BEC tensors to be symmetric
        """
        super().__init__()
        
        # Store parameters
        self.hidden_irreps = Irreps(hidden_irreps)
        self.max_ell = max_ell
        self.sh_irreps = e3nn.Irreps.spherical_harmonics(max_ell)
        self.r_max = r_max
        self.avg_num_neighbors = avg_num_neighbors
        self.num_species = num_species
        self.graph_net_steps = graph_net_steps
        self.use_sc = use_sc
        self.nonlinearities = nonlinearities
        self.radial_net_nonlinearity = radial_net_nonlinearity
        self.radial_net_n_hidden = radial_net_n_hidden
        self.radial_net_n_layers = radial_net_n_layers
        self.num_basis = num_basis
        self.scalar_mlp_std = scalar_mlp_std
        self.apply_physics_constraints = apply_physics_constraints
        self.force_symmetric_bec = force_symmetric_bec
        
        # Initialize embeddings
        self.node_embedding = LinearNodeEmbeddingBlock(
            num_species=num_species,
            irreps_out=self.hidden_irreps,
        )
        
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            avg_r_min=avg_r_min,
            basis_functions=radial_basis,
            envelope_function=radial_envelope,
        )

    def _predict_bec_tensors(self, h_node: IrrepsArray) -> jnp.ndarray:
        """
        Predict BEC tensors - can be symmetric or asymmetric.
        
        Args:
            h_node: Node features from message passing
            
        Returns:
            BEC tensors [n_nodes, 3, 3]
        """
        if self.force_symmetric_bec:
            # Symmetric case: predict 6 components
            bec_components = Linear(irreps_out="6x0e")(h_node).array  # [n_nodes, 6]
            
            # Reconstruct symmetric 3x3 tensors
            n_nodes = bec_components.shape[0]
            bec_tensors = jnp.zeros((n_nodes, 3, 3))
            
            # Fill symmetric tensor from 6 components
            # Order: [xx, yy, zz, xy, xz, yz]
            bec_tensors = bec_tensors.at[:, 0, 0].set(bec_components[:, 0])  # xx
            bec_tensors = bec_tensors.at[:, 1, 1].set(bec_components[:, 1])  # yy
            bec_tensors = bec_tensors.at[:, 2, 2].set(bec_components[:, 2])  # zz
            bec_tensors = bec_tensors.at[:, 0, 1].set(bec_components[:, 3])  # xy
            bec_tensors = bec_tensors.at[:, 1, 0].set(bec_components[:, 3])  # yx = xy
            bec_tensors = bec_tensors.at[:, 0, 2].set(bec_components[:, 4])  # xz
            bec_tensors = bec_tensors.at[:, 2, 0].set(bec_components[:, 4])  # zx = xz
            bec_tensors = bec_tensors.at[:, 1, 2].set(bec_components[:, 5])  # yz
            bec_tensors = bec_tensors.at[:, 2, 1].set(bec_components[:, 5])  # zy = yz
        else:
            # Asymmetric case: predict all 9 components
            bec_components = Linear(irreps_out="9x0e")(h_node).array  # [n_nodes, 9]
            
            # Reconstruct full 3x3 tensors
            n_nodes = bec_components.shape[0]
            
            # Reshape directly to 3x3 tensors
            # Order: [xx, xy, xz, yx, yy, yz, zx, zy, zz]
            bec_tensors = bec_components.reshape(n_nodes, 3, 3)
        
        return bec_tensors

    def _decompose_bec_tensors(self, bec_tensors: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Decompose BEC tensors into symmetric and antisymmetric parts.
        
        Args:
            bec_tensors: BEC tensors [n_nodes, 3, 3]
            
        Returns:
            Tuple of (symmetric_part [n_nodes, 3, 3], antisymmetric_part [n_nodes, 3, 3])
        """
        # Symmetric part: (T + T^T) / 2
        symmetric_part = 0.5 * (bec_tensors + jnp.swapaxes(bec_tensors, -2, -1))
        
        # Antisymmetric part: (T - T^T) / 2
        antisymmetric_part = 0.5 * (bec_tensors - jnp.swapaxes(bec_tensors, -2, -1))
        
        return symmetric_part, antisymmetric_part

    def _apply_bec_constraints(self, bec_tensors: jnp.ndarray) -> jnp.ndarray:
        """
        Apply physics constraints to BEC tensors.
        
        Args:
            bec_tensors: Raw BEC predictions [n_nodes, 3, 3]
            
        Returns:
            Constrained BEC tensors [n_nodes, 3, 3]
        """
        if not self.apply_physics_constraints:
            return bec_tensors
            
        # Apply acoustic sum rule: sum of BECs should be zero
        # This applies to both symmetric and antisymmetric parts
        bec_mean = jnp.mean(bec_tensors, axis=0, keepdims=True)  # [1, 3, 3]
        bec_constrained = bec_tensors - bec_mean  # [n_nodes, 3, 3]
        
        # Additional constraint: For cubic crystals, you might want to enforce
        # that the antisymmetric part has certain symmetries
        # This is material-specific and can be added here if needed
        
        return bec_constrained

    def _predict_dielectric_tensor(self, h_node: IrrepsArray, 
                                   node_species: jnp.ndarray) -> jnp.ndarray:
        """
        Predict dielectric tensor as a graph-level property.
        Note: Dielectric tensor is always symmetric due to thermodynamic constraints.
        
        Args:
            h_node: Node features from message passing
            node_species: Atomic species for each node
            
        Returns:
            Dielectric tensor [3, 3] (symmetric)
        """
        # Predict per-node contributions to dielectric tensor (always symmetric)
        eps_components = Linear(irreps_out="6x0e")(h_node).array  # [n_nodes, 6]
        
        # Reconstruct symmetric tensors per node
        n_nodes = eps_components.shape[0]
        eps_per_node = jnp.zeros((n_nodes, 3, 3))
        
        # Fill symmetric tensor from 6 components
        eps_per_node = eps_per_node.at[:, 0, 0].set(eps_components[:, 0])  # xx
        eps_per_node = eps_per_node.at[:, 1, 1].set(eps_components[:, 1])  # yy
        eps_per_node = eps_per_node.at[:, 2, 2].set(eps_components[:, 2])  # zz
        eps_per_node = eps_per_node.at[:, 0, 1].set(eps_components[:, 3])  # xy
        eps_per_node = eps_per_node.at[:, 1, 0].set(eps_components[:, 3])  # yx = xy
        eps_per_node = eps_per_node.at[:, 0, 2].set(eps_components[:, 4])  # xz
        eps_per_node = eps_per_node.at[:, 2, 0].set(eps_components[:, 4])  # zx = xz
        eps_per_node = eps_per_node.at[:, 1, 2].set(eps_components[:, 5])  # yz
        eps_per_node = eps_per_node.at[:, 2, 1].set(eps_components[:, 5])  # zy = yz
        
        # Aggregate to graph-level: average over nodes
        eps_graph = jnp.mean(eps_per_node, axis=0)  # [3, 3]
        
        # Ensure positive definiteness by adding small identity
        eps_regularized = eps_graph + 0.01 * jnp.eye(3)
        
        return eps_regularized

    def __call__(
        self,
        vectors: jnp.ndarray,  # [n_edges, 3]
        node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[Tuple[jnp.ndarray, jnp.ndarray]]]:
        """
        Forward pass of the model.
        
        Args:
            vectors: Edge displacement vectors [n_edges, 3]
            node_specie: Atomic species indices [n_nodes]
            senders: Source node indices [n_edges]
            receivers: Target node indices [n_edges]
            
        Returns:
            Tuple of (BEC tensors [n_nodes, 3, 3], 
                     Dielectric tensor [3, 3],
                     Denoising vectors [n_nodes, 3],
                     Optional BEC decomposition (symmetric, antisymmetric) if not forced symmetric)
        """
        r_max = jnp.float32(self.r_max)
        hidden_irreps = self.hidden_irreps
        
        # Node attribute embedding
        node_attrs = self.node_embedding(node_specie).astype(vectors.dtype)
        
        # Edge features
        lengths = safe_norm(vectors, axis=-1)
        edge_sh = e3nn.spherical_harmonics(
            self.sh_irreps,
            vectors / lengths[..., None],
            normalize=False,
            normalization="component"
        )
        embedded_dr_edge = self.radial_embedding(lengths).array
        
        # Initial node embedding
        h_node = Linear(irreps_out=hidden_irreps)(node_attrs)

        # Message passing layers
        for _ in range(self.graph_net_steps):
            h_node = NequIPConvolution(
                hidden_irreps=hidden_irreps,
                use_sc=self.use_sc,
                nonlinearities=self.nonlinearities,
                radial_net_nonlinearity=self.radial_net_nonlinearity,
                radial_net_n_hidden=self.radial_net_n_hidden,
                radial_net_n_layers=self.radial_net_n_layers,
                num_basis=self.num_basis,
                avg_num_neighbors=self.avg_num_neighbors,
                scalar_mlp_std=self.scalar_mlp_std
            )(h_node, node_attrs, edge_sh, senders, receivers, embedded_dr_edge)
        
        # Predict tensorial properties
        bec_tensors = self._predict_bec_tensors(h_node)
        bec_tensors = self._apply_bec_constraints(bec_tensors)
        
        # Decompose BEC tensors if they're asymmetric
        bec_decomposition = None
        if not self.force_symmetric_bec:
            bec_decomposition = self._decompose_bec_tensors(bec_tensors)
        
        eps_tensor = self._predict_dielectric_tensor(h_node, node_specie)
        
        # Denoising prediction (for auxiliary loss)
        h_node_denoising = Linear(irreps_out=Irreps('1x1o'))(h_node).array

        # Only return bec_decomposition if not forced symmetric
        if not self.force_symmetric_bec:
            return bec_tensors, eps_tensor, h_node_denoising, bec_decomposition
        else:
            return bec_tensors, eps_tensor, h_node_denoising, None


def BECS_EPS_model(
    *,
    r_max: float,
    train_graphs: List[jraph.GraphsTuple] = None,
    initialize_seed: Optional[int] = None,
    avg_num_neighbors: float = "average",
    avg_r_min: float = None,
    num_species: int = None,
    path_normalization="path",
    gradient_normalization="path",
    radial_basis: Callable[[jnp.ndarray], jnp.ndarray] = bessel_basis,
    radial_envelope: Callable[[jnp.ndarray], jnp.ndarray] = soft_envelope,
    save_dir_name=None,
    reload=None,
    apply_physics_constraints: bool = True,
    force_symmetric_bec: bool = False,  # NEW: Control BEC symmetry
    **kwargs,
):
    """
    Create improved BECS/EPS model with physics constraints.
    
    Args:
        r_max: Cutoff radius
        train_graphs: Training data for setup
        initialize_seed: Random seed for initialization
        avg_num_neighbors: Average number of neighbors
        avg_r_min: Average minimum distance
        num_species: Number of atomic species
        path_normalization: e3nn path normalization
        gradient_normalization: e3nn gradient normalization
        radial_basis: Radial basis function
        radial_envelope: Radial envelope function
        save_dir_name: Directory to save model setup
        reload: Directory to reload model setup from
        apply_physics_constraints: Apply BEC physics constraints
        force_symmetric_bec: Force BEC tensors to be symmetric (default: False for asymmetric)
        **kwargs: Additional model parameters
        
    Returns:
        Tuple of (model_apply_fn, params, num_message_passing_steps)
    """
    if reload is None:
        becs_eps_model_setup = {}
        
        if train_graphs is None:
            z_table = None
        else:
            z_table = get_atomic_number_table_from_zs(
                z for graph in train_graphs for z in graph.nodes.species
            )
        print(f"z_table= {z_table}")
        
        becs_eps_model_setup['z_table'] = z_table

        if avg_num_neighbors == "average":
            avg_num_neighbors = compute_avg_num_neighbors(train_graphs)
            print(f"Compute the average number of neighbors: {avg_num_neighbors:.3f}")
        else:
            print(f"Use the average number of neighbors: {avg_num_neighbors:.3f}")
            
        becs_eps_model_setup['avg_num_neighbors'] = avg_num_neighbors

        if avg_r_min == "average":
            avg_r_min = None  # placeholder
            print("Do not normalize the radial basis (avg_r_min=None)")
        elif avg_r_min is None:
            print("Do not normalize the radial basis (avg_r_min=None)")
        else:
            print(f"Use the average min neighbor distance: {avg_r_min:.3f}")

        becs_eps_model_setup['avg_r_min'] = avg_r_min
        
        if save_dir_name and str(save_dir_name).strip():
            try:
                save_path = str(save_dir_name).replace(':', '-')
                os.makedirs(save_path, exist_ok=True)
                save_file = os.path.join(save_path, "becs_eps_model_setup.pkl")
                with open(save_file, "wb") as f:
                    pickle.dump(becs_eps_model_setup, f)
            except Exception as e:
                print(f"Warning: Could not save model setup: {e}")
    else:
        try:
            reload_path = str(reload).replace(':', '-')
            reload_file = os.path.join(reload_path, "becs_eps_model_setup.pkl")
            with open(reload_file, "rb") as f:
                becs_eps_model_setup = pickle.load(f)
                
            if save_dir_name and str(save_dir_name).strip():
                try:
                    save_path = str(save_dir_name).replace(':', '-')
                    os.makedirs(save_path, exist_ok=True)
                    save_file = os.path.join(save_path, "becs_eps_model_setup.pkl")
                    with open(save_file, "wb") as f:
                        pickle.dump(becs_eps_model_setup, f)
                except Exception as e:
                    print(f"Warning: Could not save model setup: {e}")
        except Exception as e:
            print(f"Error loading model setup: {e}")
            raise
        
        z_table = becs_eps_model_setup['z_table']
        avg_num_neighbors = becs_eps_model_setup['avg_num_neighbors']
        avg_r_min = becs_eps_model_setup['avg_r_min']

    # Validate num_species consistency
    if z_table is not None and max(z_table.zs) >= num_species:
        raise ValueError(f"max(z_table.zs)={max(z_table.zs)} >= num_species={num_species}")

    # Update kwargs with computed values
    kwargs.update(
        dict(
            r_max=r_max,
            avg_num_neighbors=avg_num_neighbors,
            avg_r_min=avg_r_min,
            num_species=num_species,
            radial_basis=radial_basis,
            radial_envelope=radial_envelope,
            apply_physics_constraints=apply_physics_constraints,
            force_symmetric_bec=force_symmetric_bec,
        )
    )
    # Ensure required kwargs for BECS_EPS_nequip_base_Model are present
    if 'use_sc' not in kwargs:
        kwargs['use_sc'] = True
    if 'nonlinearities' not in kwargs:
        kwargs['nonlinearities'] = {'e': 'swish', 'o': 'tanh'}
    
    symmetry_info = "symmetric" if force_symmetric_bec else "asymmetric"
    print(f"Create BECS/EPS (NequIP-based) model with {symmetry_info} BEC tensors and parameters {kwargs}")

    @hk.without_apply_rng
    @hk.transform
    def model_(
        vectors: jnp.ndarray,  # [n_edges, 3]
        node_z: jnp.ndarray,  # [n_nodes]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[Tuple[jnp.ndarray, jnp.ndarray]]]:
        e3nn.config("path_normalization", path_normalization)
        e3nn.config("gradient_normalization", gradient_normalization)
        
        becs_eps = BECS_EPS_nequip_base_Model(**kwargs)

        if hk.running_init():
            print(f"model: hidden_irreps={becs_eps.hidden_irreps} "
                  f"sh_irreps={becs_eps.sh_irreps}")
            print(f"BEC tensor mode: {'symmetric (6 components)' if force_symmetric_bec else 'asymmetric (9 components)'}")

        return becs_eps(vectors, node_z, senders, receivers)

    if initialize_seed is not None and reload is None:
        params = jax.jit(model_.init)(
            jax.random.PRNGKey(initialize_seed),
            jnp.zeros((1, 3)),
            jnp.array([16]),
            jnp.array([0]),
            jnp.array([0]),
        )
    elif reload is not None:
        with open(f"{reload}/params.pkl", "rb") as f:
            params = pickle.load(f)
    else:
        params = None

    return model_.apply, params, kwargs.get('graph_net_steps', 3)