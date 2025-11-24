"""
Simplified aircraft model for BVR combat simulation
Focus on detection, missiles, and tactics - not detailed flight dynamics
"""
import numpy as np
from typing import Optional, List
from ..utils import Vector3


class Aircraft:
    """
    Simplified aircraft model for BVR combat

    No complex aerodynamics - just position, velocity, and turn performance
    """

    def __init__(self, name: str, position: Vector3, velocity: Vector3):
        self.name = name
        self.position = position
        self.velocity = velocity

        # Performance characteristics (to be set by subclasses)
        self.max_speed = 600.0  # m/s (Mach 2 class)
        self.cruise_speed = 250.0  # m/s
        self.max_turn_rate = np.radians(20)  # rad/s (sustained turn rate)
        self.max_acceleration = 50.0  # m/s²

        # Radar Cross Section (m²) - aspect dependent
        self.rcs_frontal = 5.0  # Frontal RCS
        self.rcs_side = 10.0  # Side RCS
        self.rcs_rear = 15.0  # Rear RCS

        # Sensor characteristics
        self.radar_power = 1000.0  # Relative radar power (arbitrary units)
        self.radar_gain = 40.0  # dB
        self.radar_frequency = 10e9  # Hz (X-band typical)

        # Weapons
        self.missiles: List = []  # Available missiles
        self.max_missiles = 6

        # State
        self.alive = True
        self.detected_targets: dict = {}  # target_id -> detection_time
        self.tracking_targets: set = set()  # Targets with lock

    def get_speed(self) -> float:
        """Get current speed in m/s"""
        return self.velocity.magnitude()

    def get_altitude(self) -> float:
        """Get altitude in meters (negative Z)"""
        return -self.position.z

    def get_heading(self) -> Vector3:
        """Get current heading as unit vector"""
        return self.velocity.normalized()

    def get_rcs(self, observer_pos: Vector3) -> float:
        """
        Get RCS as seen from observer position (aspect-dependent)

        Args:
            observer_pos: Position of observing radar

        Returns:
            RCS in m²
        """
        # Line of sight from aircraft to observer
        los = (observer_pos - self.position).normalized()
        heading = self.get_heading()

        # Aspect angle (0 = head-on, 90 = side, 180 = tail)
        cos_aspect = heading.dot(los)
        aspect_angle = np.arccos(np.clip(cos_aspect, -1.0, 1.0))

        # Interpolate RCS based on aspect
        aspect_deg = np.degrees(aspect_angle)

        if aspect_deg < 30:
            # Frontal aspect
            return self.rcs_frontal
        elif aspect_deg < 150:
            # Side aspect (interpolate)
            t = (aspect_deg - 30) / 120.0
            return self.rcs_frontal + t * (self.rcs_side - self.rcs_frontal)
        else:
            # Rear aspect
            return self.rcs_rear

    def update(self, dt: float, commanded_heading: Optional[Vector3] = None,
               commanded_speed: Optional[float] = None):
        """
        Update aircraft state with simple kinematics

        Args:
            dt: Time step in seconds
            commanded_heading: Desired heading (will turn toward it)
            commanded_speed: Desired speed (will accelerate toward it)
        """
        if not self.alive:
            return

        # Speed control
        if commanded_speed is not None:
            current_speed = self.get_speed()
            speed_error = commanded_speed - current_speed

            # Accelerate/decelerate with limits
            delta_speed = np.clip(speed_error, -self.max_acceleration * dt,
                                 self.max_acceleration * dt)
            new_speed = current_speed + delta_speed

            # Update velocity magnitude
            if current_speed > 1e-6:
                self.velocity = self.velocity.normalized() * new_speed
            else:
                self.velocity = Vector3(new_speed, 0, 0)

        # Heading control
        if commanded_heading is not None:
            current_heading = self.get_heading()
            desired_heading = commanded_heading.normalized()

            # Calculate angle between current and desired heading
            cos_angle = current_heading.dot(desired_heading)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

            if angle > 0.01:  # More than ~0.5 degrees
                # Turn toward desired heading
                max_turn = self.max_turn_rate * dt

                if angle < max_turn:
                    # Can reach desired heading this timestep
                    turn_fraction = 1.0
                else:
                    # Turn maximum amount
                    turn_fraction = max_turn / angle

                # Spherical interpolation (simplified)
                new_heading = (current_heading * (1 - turn_fraction) +
                             desired_heading * turn_fraction).normalized()

                # Update velocity direction
                speed = self.get_speed()
                self.velocity = new_heading * speed

        # Update position
        self.position = self.position + self.velocity * dt

    def can_detect(self, target: 'Aircraft', weather_factor: float = 1.0) -> bool:
        """
        Determine if this aircraft's radar can detect target

        Uses simplified radar equation

        Args:
            target: Target aircraft
            weather_factor: Weather degradation (1.0 = clear, 0.5 = poor)

        Returns:
            True if target is detected
        """
        if not self.alive or not target.alive:
            return False

        # Range to target
        range_m = self.position.distance_to(target.position)

        # Target RCS from our perspective
        target_rcs = target.get_rcs(self.position)

        # Simplified radar equation: max_range ∝ RCS^0.25
        # Base detection range against 1m² target at sea level
        base_range = 100000  # 100 km baseline

        # Actual max range scales with RCS^0.25
        max_range = base_range * (target_rcs ** 0.25) * weather_factor

        # Add some probabilistic detection near the edge
        if range_m > max_range * 1.1:
            return False
        elif range_m < max_range * 0.9:
            return True
        else:
            # Probabilistic detection in edge zone
            prob = 1.0 - (range_m - max_range * 0.9) / (max_range * 0.2)
            return np.random.random() < prob

    def get_detection_range(self, target_rcs: float) -> float:
        """
        Calculate detection range against a target with given RCS

        Args:
            target_rcs: Target RCS in m²

        Returns:
            Detection range in meters
        """
        base_range = 100000  # 100 km against 1 m² target
        return base_range * (target_rcs ** 0.25)

    def kill(self):
        """Mark aircraft as destroyed"""
        self.alive = False
        self.velocity = Vector3(0, 0, 0)

    def __repr__(self):
        status = "ALIVE" if self.alive else "DEAD"
        alt_ft = -self.position.z / 0.3048
        speed_kts = self.get_speed() / 0.514444
        return f"{self.name} [{status}] Alt:{alt_ft:.0f}ft Speed:{speed_kts:.0f}kts"
