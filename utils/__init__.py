"""Utils package for drifts-and-complexity project.

This package contains utility modules for:
- Complexity analysis and metrics
- Drift detection and characterization
- Data processing and normalization
- Plotting and visualization
- Population analysis and bootstrapping
- Windowing and sampling utilities
"""

from . import constants
from . import helpers
from . import drift_io
from . import sampling_helper
from . import parallel

__version__ = "0.1.0"
__all__ = [
    "constants",
    "helpers", 
    "drift_io",
    "sampling_helper",
    "parallel",
]
