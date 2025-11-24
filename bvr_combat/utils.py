"""
Utility functions for BVR combat simulation
"""
import numpy as np
from typing import Tuple


class Vector3:
    """Simple 3D vector for position and velocity"""

    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def magnitude(self) -> float:
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self):
        mag = self.magnitude()
        if mag < 1e-10:
            return Vector3(1, 0, 0)
        return self / mag

    def dot(self, other) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def distance_to(self, other) -> float:
        return (self - other).magnitude()

    def __repr__(self):
        return f"Vector3({self.x:.1f}, {self.y:.1f}, {self.z:.1f})"


def meters_to_km(meters: float) -> float:
    """Convert meters to kilometers"""
    return meters / 1000.0


def km_to_meters(km: float) -> float:
    """Convert kilometers to meters"""
    return km * 1000.0


def feet_to_meters(feet: float) -> float:
    """Convert feet to meters"""
    return feet * 0.3048


def meters_to_feet(meters: float) -> float:
    """Convert meters to feet"""
    return meters / 0.3048


def knots_to_mps(knots: float) -> float:
    """Convert knots to meters per second"""
    return knots * 0.514444


def mps_to_knots(mps: float) -> float:
    """Convert meters per second to knots"""
    return mps / 0.514444


def get_aspect_angle(pos_observer: Vector3, pos_target: Vector3,
                     vel_target: Vector3) -> float:
    """
    Calculate aspect angle: angle between target's velocity and line-of-sight

    Returns angle in radians (0 = head-on, π = tail-on)
    """
    los = pos_observer - pos_target  # Line of sight from target to observer

    if los.magnitude() < 1e-6 or vel_target.magnitude() < 1e-6:
        return 0.0

    los_norm = los.normalized()
    vel_norm = vel_target.normalized()

    cos_aspect = los_norm.dot(vel_norm)
    return np.arccos(np.clip(cos_aspect, -1.0, 1.0))


def get_closure_rate(pos1: Vector3, vel1: Vector3,
                     pos2: Vector3, vel2: Vector3) -> float:
    """
    Calculate closure rate between two aircraft

    Returns rate in m/s (positive = closing, negative = opening)
    """
    relative_pos = pos2 - pos1
    relative_vel = vel2 - vel1

    range_val = relative_pos.magnitude()
    if range_val < 1e-6:
        return 0.0

    # Rate of change of range
    closure = -relative_vel.dot(relative_pos) / range_val
    return closure


def calculate_intercept_point(shooter_pos: Vector3, shooter_vel: Vector3,
                              target_pos: Vector3, target_vel: Vector3,
                              missile_speed: float) -> Tuple[Vector3, float]:
    """
    Calculate intercept point for proportional navigation

    Returns (intercept_point, time_to_intercept)
    """
    # Relative position and velocity
    rel_pos = target_pos - shooter_pos
    rel_vel = target_vel - shooter_vel

    # Quadratic equation: |rel_pos + rel_vel*t| = missile_speed * t
    a = rel_vel.dot(rel_vel) - missile_speed**2
    b = 2 * rel_pos.dot(rel_vel)
    c = rel_pos.dot(rel_pos)

    discriminant = b**2 - 4*a*c

    if discriminant < 0 or abs(a) < 1e-10:
        # No intercept possible, aim at current position
        return target_pos, rel_pos.magnitude() / missile_speed

    t1 = (-b + np.sqrt(discriminant)) / (2*a)
    t2 = (-b - np.sqrt(discriminant)) / (2*a)

    # Choose positive, smaller time
    t = min(t for t in [t1, t2] if t > 0) if any(t > 0 for t in [t1, t2]) else 0

    intercept = target_pos + target_vel * t
    return intercept, t
