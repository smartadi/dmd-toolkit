"""DMD toolkit — variants from the Kutz canon."""

from .exact import ExactDMD, exact_dmd
from .optimized import OptimizedDMD, optimized_dmd
from .bop import BOPDMD, bop_dmd
from .multires import MultiResolutionDMD, mr_dmd
from .havok import HAVOK, havok
from .pidmd import piDMD, pidmd
from .kernel import KernelDMD, kernel_dmd
from . import data
from . import plot
from . import utils
from . import neural

__all__ = [
    "ExactDMD",
    "exact_dmd",
    "OptimizedDMD",
    "optimized_dmd",
    "BOPDMD",
    "bop_dmd",
    "MultiResolutionDMD",
    "mr_dmd",
    "HAVOK",
    "havok",
    "piDMD",
    "pidmd",
    "KernelDMD",
    "kernel_dmd",
    "data",
    "plot",
    "utils",
    "neural",
]
