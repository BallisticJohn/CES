"""
Quaternion class for 3D rotations
Avoids gimbal lock and provides smooth interpolation
Format: q = w + xi + yj + zk
"""

import numpy as np
from typing import Tuple
from .vector import Vector3


class Quaternion:
    """Quaternion for representing 3D rotations"""

    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """Initialize quaternion (w is scalar part, x,y,z are vector part)"""
        self.data = np.array([w, x, y, z], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Quaternion':
        """Create from numpy array"""
        return cls(arr[0], arr[1], arr[2], arr[3])

    @classmethod
    def identity(cls) -> 'Quaternion':
        """Identity quaternion (no rotation)"""
        return cls(1, 0, 0, 0)

    @classmethod
    def from_axis_angle(cls, axis: Vector3, angle: float) -> 'Quaternion':
        """
        Create quaternion from axis-angle representation

        Args:
            axis: Rotation axis (will be normalized)
            angle: Rotation angle in radians
        """
        half_angle = angle * 0.5
        s = np.sin(half_angle)
        normalized = axis.normalized()
        return cls(
            np.cos(half_angle),
            normalized.x * s,
            normalized.y * s,
            normalized.z * s
        )

    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float) -> 'Quaternion':
        """
        Create quaternion from Euler angles (intrinsic rotations)

        Args:
            roll: Rotation around X-axis (radians)
            pitch: Rotation around Y-axis (radians)
            yaw: Rotation around Z-axis (radians)

        Convention: ZYX (yaw, pitch, roll)
        """
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        return cls(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        )

    @classmethod
    def from_rotation_matrix(cls, R: np.ndarray) -> 'Quaternion':
        """Create quaternion from 3x3 rotation matrix"""
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            return cls(
                0.25 / s,
                (R[2, 1] - R[1, 2]) * s,
                (R[0, 2] - R[2, 0]) * s,
                (R[1, 0] - R[0, 1]) * s
            )
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            return cls(
                (R[2, 1] - R[1, 2]) / s,
                0.25 * s,
                (R[0, 1] + R[1, 0]) / s,
                (R[0, 2] + R[2, 0]) / s
            )
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            return cls(
                (R[0, 2] - R[2, 0]) / s,
                (R[0, 1] + R[1, 0]) / s,
                0.25 * s,
                (R[1, 2] + R[2, 1]) / s
            )
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            return cls(
                (R[1, 0] - R[0, 1]) / s,
                (R[0, 2] + R[2, 0]) / s,
                (R[1, 2] + R[2, 1]) / s,
                0.25 * s
            )

    # Properties
    @property
    def w(self) -> float:
        return self.data[0]

    @property
    def x(self) -> float:
        return self.data[1]

    @property
    def y(self) -> float:
        return self.data[2]

    @property
    def z(self) -> float:
        return self.data[3]

    # Operations
    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """Quaternion multiplication (composition of rotations)"""
        w1, x1, y1, z1 = self.data
        w2, x2, y2, z2 = other.data
        return Quaternion(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        )

    def conjugate(self) -> 'Quaternion':
        """Quaternion conjugate (inverse rotation for unit quaternions)"""
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def magnitude(self) -> float:
        """Quaternion magnitude"""
        return np.linalg.norm(self.data)

    def normalized(self) -> 'Quaternion':
        """Return normalized quaternion"""
        mag = self.magnitude()
        if mag < 1e-10:
            return Quaternion.identity()
        return Quaternion.from_array(self.data / mag)

    def inverse(self) -> 'Quaternion':
        """Quaternion inverse"""
        conj = self.conjugate()
        mag_sq = np.dot(self.data, self.data)
        if mag_sq < 1e-10:
            return Quaternion.identity()
        return Quaternion.from_array(conj.data / mag_sq)

    def rotate_vector(self, v: Vector3) -> Vector3:
        """
        Rotate a vector by this quaternion
        v' = q * v * q^(-1)
        """
        # Optimized version avoiding full quaternion multiplication
        qw, qx, qy, qz = self.data
        vx, vy, vz = v.x, v.y, v.z

        # Calculate qv = q * [0, v]
        tx = qw * vx + qy * vz - qz * vy
        ty = qw * vy + qz * vx - qx * vz
        tz = qw * vz + qx * vy - qy * vx
        tw = -qx * vx - qy * vy - qz * vz

        # Calculate qv * q^(-1)
        rx = tw * -qx + tx * qw + ty * -qz - tz * -qy
        ry = tw * -qy - tx * -qz + ty * qw + tz * -qx
        rz = tw * -qz + tx * -qy - ty * -qx + tz * qw

        return Vector3(rx, ry, rz)

    def to_euler(self) -> Tuple[float, float, float]:
        """
        Convert to Euler angles (roll, pitch, yaw)
        Returns: (roll, pitch, yaw) in radians
        """
        w, x, y, z = self.data

        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def to_rotation_matrix(self) -> np.ndarray:
        """Convert to 3x3 rotation matrix"""
        w, x, y, z = self.data

        return np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])

    def slerp(self, other: 'Quaternion', t: float) -> 'Quaternion':
        """
        Spherical linear interpolation

        Args:
            other: Target quaternion
            t: Interpolation parameter [0, 1]
        """
        dot = np.dot(self.data, other.data)

        # If negative dot, negate one quaternion to take shorter path
        if dot < 0:
            other_data = -other.data
            dot = -dot
        else:
            other_data = other.data

        # If very close, use linear interpolation
        if dot > 0.9995:
            result = self.data + t * (other_data - self.data)
            return Quaternion.from_array(result / np.linalg.norm(result))

        # Calculate spherical interpolation
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta = np.sin(theta)

        a = np.sin((1.0 - t) * theta) / sin_theta
        b = np.sin(t * theta) / sin_theta

        result = a * self.data + b * other_data
        return Quaternion.from_array(result)

    def clone(self) -> 'Quaternion':
        """Create a copy"""
        return Quaternion.from_array(self.data.copy())

    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return self.data.copy()

    def __repr__(self) -> str:
        return f"Quaternion(w={self.w:.6f}, x={self.x:.6f}, y={self.y:.6f}, z={self.z:.6f})"

    def __str__(self) -> str:
        return f"Q({self.w:.3f} + {self.x:.3f}i + {self.y:.3f}j + {self.z:.3f}k)"

    def __eq__(self, other: 'Quaternion') -> bool:
        return np.allclose(self.data, other.data)
