"""Core physics engine for flight simulation"""

from .vector import Vector3
from .quaternion import Quaternion
from .atmosphere import Atmosphere

__all__ = ['Vector3', 'Quaternion', 'Atmosphere']
