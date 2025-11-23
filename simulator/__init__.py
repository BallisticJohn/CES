"""
Air Combat Simulator - F-86 Sabre vs MiG-15
High-fidelity physics-based combat simulation
"""

__version__ = "0.1.0"

from .aircraft.f86 import F86Sabre
from .aircraft.mig15 import MiG15
from .simulation.engagement import EngagementSimulation
from .simulation.monte_carlo import MonteCarloRunner

__all__ = [
    'F86Sabre',
    'MiG15',
    'EngagementSimulation',
    'MonteCarloRunner',
]
