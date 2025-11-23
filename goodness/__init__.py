"""
Goodness functions package.
Auto-imports all goodness function modules to register them.
"""

from .registry import (
    register,
    get_goodness_function,
    list_available_goodness_functions,
    is_registered
)

# Import all goodness function modules to trigger registration
from . import g_sum_of_squares  # Baseline from original paper
from . import g_sparse_l1_local
from . import g_hebbian_local
from . import g_info_nce_local
#from . import g_info_nce_proj_local
from . import g_nt_xent_local
from . import g_decorrelation_local
from . import g_gaussian_energy_local
from . import g_huber_norm_local
from . import g_l2_normalized_energy_local
from . import g_oja_local
from . import g_outlier_trimmed_energy_local
from . import g_pca_energy_local
from . import g_softmax_energy_margin_local
#from . import g_temperature_annealed_energy_local
from . import g_tempered_energy_local
from . import g_whitened_energy_local
from . import g_bcm_local

# New goodness functions from different domains
from . import g_attention_weighted_local
from . import g_game_theoretic_local
from . import g_fractal_dimension_local
from . import g_predictive_coding_local
from . import g_triplet_margin_local

__all__ = [
    'register',
    'get_goodness_function',
    'list_available_goodness_functions',
    'is_registered'
]
