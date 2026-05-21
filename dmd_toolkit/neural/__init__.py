"""Neural-network DMD/Koopman variants from recent Kutz-group work.

All models here require torch. Imports are lazy so the rest of dmd_toolkit
stays usable without torch installed.
"""

from .shred import SHRED, fit_shred, predict_shred, make_sensor_lag_dataset
from .dpk import DeepProbKoopman, fit_dpk
from .discrepancy import DiscrepancyModel
from .sks import SINDyKoopmanSHRED, fit_sks
from .manifold import KoopmanAutoencoder, fit_koopman_ae

__all__ = [
    "SHRED",
    "fit_shred",
    "predict_shred",
    "make_sensor_lag_dataset",
    "DeepProbKoopman",
    "fit_dpk",
    "DiscrepancyModel",
    "SINDyKoopmanSHRED",
    "fit_sks",
    "KoopmanAutoencoder",
    "fit_koopman_ae",
]
