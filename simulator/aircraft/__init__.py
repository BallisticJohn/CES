"""Aircraft models and implementations"""

from .base import Aircraft
from .f86 import F86Sabre
from .mig15 import MiG15

__all__ = ['Aircraft', 'F86Sabre', 'MiG15']
