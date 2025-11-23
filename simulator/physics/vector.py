"""
3D Vector operations using NumPy
Coordinate system: Right-handed NED (North-East-Down) for world frame
                    X-forward, Y-right, Z-down for body frame
"""

import numpy as np
from typing import Union


class Vector3:
    """3D vector with common operations for flight simulation"""

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """Initialize vector from components or array"""
        if isinstance(x, (list, tuple, np.ndarray)):
            self.data = np.array(x, dtype=np.float64)
        else:
            self.data = np.array([x, y, z], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Vector3':
        """Create vector from numpy array"""
        return cls(arr[0], arr[1], arr[2])

    @classmethod
    def zero(cls) -> 'Vector3':
        """Zero vector"""
        return cls(0, 0, 0)

    @classmethod
    def forward(cls) -> 'Vector3':
        """Forward unit vector (X-axis)"""
        return cls(1, 0, 0)

    @classmethod
    def right(cls) -> 'Vector3':
        """Right unit vector (Y-axis)"""
        return cls(0, 1, 0)

    @classmethod
    def down(cls) -> 'Vector3':
        """Down unit vector (Z-axis)"""
        return cls(0, 0, 1)

    @classmethod
    def up(cls) -> 'Vector3':
        """Up unit vector (-Z-axis)"""
        return cls(0, 0, -1)

    # Properties
    @property
    def x(self) -> float:
        return self.data[0]

    @x.setter
    def x(self, value: float):
        self.data[0] = value

    @property
    def y(self) -> float:
        return self.data[1]

    @y.setter
    def y(self, value: float):
        self.data[1] = value

    @property
    def z(self) -> float:
        return self.data[2]

    @z.setter
    def z(self, value: float):
        self.data[2] = value

    # Arithmetic operations
    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3.from_array(self.data + other.data)

    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3.from_array(self.data - other.data)

    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3.from_array(self.data * scalar)

    def __rmul__(self, scalar: float) -> 'Vector3':
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> 'Vector3':
        return Vector3.from_array(self.data / scalar)

    def __neg__(self) -> 'Vector3':
        return Vector3.from_array(-self.data)

    # Vector operations
    def dot(self, other: 'Vector3') -> float:
        """Dot product"""
        return np.dot(self.data, other.data)

    def cross(self, other: 'Vector3') -> 'Vector3':
        """Cross product"""
        return Vector3.from_array(np.cross(self.data, other.data))

    def magnitude(self) -> float:
        """Vector magnitude (length)"""
        return np.linalg.norm(self.data)

    def magnitude_squared(self) -> float:
        """Squared magnitude (faster than magnitude)"""
        return np.dot(self.data, self.data)

    def normalized(self) -> 'Vector3':
        """Return normalized (unit) vector"""
        mag = self.magnitude()
        if mag < 1e-10:
            return Vector3.zero()
        return self / mag

    def distance_to(self, other: 'Vector3') -> float:
        """Distance to another vector"""
        return (self - other).magnitude()

    def angle_to(self, other: 'Vector3') -> float:
        """Angle to another vector in radians"""
        dot = self.dot(other)
        mag = self.magnitude() * other.magnitude()
        if mag < 1e-10:
            return 0.0
        return np.arccos(np.clip(dot / mag, -1.0, 1.0))

    def component_multiply(self, other: 'Vector3') -> 'Vector3':
        """Component-wise multiplication"""
        return Vector3.from_array(self.data * other.data)

    def lerp(self, other: 'Vector3', t: float) -> 'Vector3':
        """Linear interpolation"""
        return self + (other - self) * t

    def project_onto(self, other: 'Vector3') -> 'Vector3':
        """Project this vector onto another"""
        other_mag_sq = other.magnitude_squared()
        if other_mag_sq < 1e-10:
            return Vector3.zero()
        return other * (self.dot(other) / other_mag_sq)

    def reflect(self, normal: 'Vector3') -> 'Vector3':
        """Reflect vector about a normal"""
        return self - normal * (2.0 * self.dot(normal))

    # Utility methods
    def clone(self) -> 'Vector3':
        """Create a copy"""
        return Vector3.from_array(self.data.copy())

    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return self.data.copy()

    def __repr__(self) -> str:
        return f"Vector3({self.x:.6f}, {self.y:.6f}, {self.z:.6f})"

    def __str__(self) -> str:
        return f"({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

    def __eq__(self, other: 'Vector3') -> bool:
        return np.allclose(self.data, other.data)

    def __getitem__(self, index: int) -> float:
        return self.data[index]

    def __setitem__(self, index: int, value: float):
        self.data[index] = value
