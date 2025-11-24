"""
Air-to-air missile models for BVR combat
"""
import numpy as np
from typing import Optional
from ..utils import Vector3, calculate_intercept_point


class Missile:
    """
    Base class for air-to-air missiles

    Simplified kinematics with energy model
    """

    def __init__(self, name: str, launch_pos: Vector3, launch_vel: Vector3,
                 target_id: str):
        self.name = name
        self.position = launch_pos
        self.velocity = launch_vel
        self.target_id = target_id

        # Performance characteristics (to be set by subclasses)
        self.max_speed = 1200.0  # m/s (Mach 3.5+)
        self.boost_acceleration = 400.0  # m/s² during boost phase
        self.boost_time = 5.0  # seconds
        self.max_range = 100000  # meters (kinematic range)
        self.no_escape_range = 30000  # meters (target can't outrun)
        self.max_g = 40.0  # Maximum g-load for maneuvering

        # State
        self.active = True
        self.time_of_flight = 0.0
        self.fuel_remaining = 1.0  # 0.0 to 1.0
        self.guidance_active = False  # Terminal guidance active
        self.seeker_lock = False

        # Energy model
        self.energy = 0.0  # Specific energy (v²/2 + gh)

    def update(self, dt: float, target_pos: Vector3, target_vel: Vector3):
        """
        Update missile state with proportional navigation

        Args:
            dt: Time step
            target_pos: Current target position
            target_vel: Current target velocity
        """
        if not self.active:
            return

        self.time_of_flight += dt

        # Boost phase
        if self.time_of_flight < self.boost_time:
            # Accelerate toward max speed
            current_speed = self.velocity.magnitude()
            if current_speed < self.max_speed:
                accel = min(self.boost_acceleration,
                          (self.max_speed - current_speed) / dt)

                # Accelerate in direction of target
                to_target = (target_pos - self.position).normalized()
                self.velocity = self.velocity + to_target * accel * dt

            self.fuel_remaining = 1.0 - (self.time_of_flight / self.boost_time)
        else:
            # Coast phase - gradually lose energy due to drag
            self.fuel_remaining = 0.0

            # Simple drag model: deceleration proportional to v²
            speed = self.velocity.magnitude()
            drag_decel = 0.5 * speed / 100.0  # Simplified
            new_speed = max(speed - drag_decel * dt, 200.0)  # Minimum speed

            if speed > 1e-6:
                self.velocity = self.velocity.normalized() * new_speed

        # Proportional navigation guidance
        if self.guidance_active:
            # Calculate intercept point
            intercept, time_to_intercept = calculate_intercept_point(
                self.position, self.velocity,
                target_pos, target_vel,
                self.velocity.magnitude()
            )

            # Steer toward intercept
            to_intercept = (intercept - self.position).normalized()
            current_heading = self.velocity.normalized()

            # Calculate required turn rate
            angle_to_intercept = np.arccos(
                np.clip(current_heading.dot(to_intercept), -1.0, 1.0)
            )

            if angle_to_intercept > 0.01:  # More than ~0.5 degrees
                # Maximum turn rate based on current speed and max g
                speed = self.velocity.magnitude()
                max_turn_rate = (self.max_g * 9.81) / speed  # rad/s

                turn_this_step = min(max_turn_rate * dt, angle_to_intercept)
                turn_fraction = turn_this_step / angle_to_intercept

                # Update heading
                new_heading = (current_heading * (1 - turn_fraction) +
                             to_intercept * turn_fraction).normalized()
                self.velocity = new_heading * speed

        # Update position
        self.position = self.position + self.velocity * dt

        # Update energy state
        altitude = -self.position.z
        speed = self.velocity.magnitude()
        self.energy = (speed ** 2) / 2.0 + 9.81 * altitude

    def check_proximity(self, target_pos: Vector3, target_vel: Vector3,
                       prev_missile_pos: Vector3) -> Optional[float]:
        """
        Check if missile is close enough to detonate

        Checks both current distance AND closest approach during last timestep
        to avoid skipping past target at high speeds

        Returns:
            Probability of kill if within lethal radius, None otherwise
        """
        current_distance = self.position.distance_to(target_pos)

        # Proximity fuse activation radius
        # Larger radius to account for high-speed intercepts
        fuse_radius = 50.0  # meters (increased from 20m)

        # Also check closest approach during the timestep
        # This prevents missiles from skipping past targets
        missile_movement = self.position - prev_missile_pos
        to_target_start = target_pos - prev_missile_pos

        # Project target onto missile's movement vector
        if missile_movement.magnitude() > 1e-6:
            t = to_target_start.dot(missile_movement) / (missile_movement.magnitude() ** 2)
            t = np.clip(t, 0.0, 1.0)  # Clamp to this timestep

            closest_point = prev_missile_pos + missile_movement * t
            closest_distance = closest_point.distance_to(target_pos)
        else:
            closest_distance = current_distance

        # Use the smaller of current or closest approach
        miss_distance = min(current_distance, closest_distance)

        if miss_distance < fuse_radius:
            # Calculate Pk based on miss distance
            # Closer = higher Pk
            pk_base = 0.85  # Base Pk for direct hit
            pk_miss = pk_base * (1.0 - (miss_distance / fuse_radius) ** 2)

            # Boost Pk for high closing speed (better fragmentation)
            target_speed = target_vel.magnitude()
            closing_speed = self.velocity.magnitude() + target_speed
            if closing_speed > 800:  # Very high speed intercept
                pk_miss *= 1.15

            return np.clip(pk_miss, 0.0, 0.95)

        return None

    def is_active(self) -> bool:
        """Check if missile is still active"""
        # Missile becomes inactive after timeout or out of energy
        max_tof = self.max_range / 400.0  # Conservative time limit

        if self.time_of_flight > max_tof:
            self.active = False

        return self.active

    def __repr__(self):
        status = "ACTIVE" if self.active else "INACTIVE"
        tof = self.time_of_flight
        speed = self.velocity.magnitude()
        return f"{self.name} [{status}] ToF:{tof:.1f}s Speed:{speed:.0f}m/s"


class AIM120C(Missile):
    """
    AIM-120C AMRAAM (Advanced Medium-Range Air-to-Air Missile)

    Primary BVR weapon of F-16, F-15, F-22, F-35
    """

    def __init__(self, launch_pos: Vector3, launch_vel: Vector3,
                 target_id: str, name: str = "AIM-120C"):
        super().__init__(name, launch_pos, launch_vel, target_id)

        # AIM-120C performance
        self.max_speed = 1200.0  # m/s (~Mach 4 at altitude)
        self.boost_acceleration = 400.0  # High thrust motor
        self.boost_time = 4.0  # Seconds of powered flight
        self.max_range = 105000  # 105 km (optimal conditions)
        self.no_escape_range = 35000  # 35 km (conservative)
        self.max_g = 40.0  # 40g maneuvering capability

        # Active radar homing - goes autonomous after midcourse
        self.midcourse_distance = 15000  # 15 km - goes active
        self.guidance_active = True  # Can start guiding immediately

        # Warhead
        self.warhead_weight = 22.0  # kg
        self.lethal_radius = 15.0  # meters


class R77(Missile):
    """
    R-77 (AA-12 Adder) - Russian medium-range AAM

    Primary BVR weapon of MiG-29, Su-27, Su-35
    """

    def __init__(self, launch_pos: Vector3, launch_vel: Vector3,
                 target_id: str, name: str = "R-77"):
        super().__init__(name, launch_pos, launch_vel, target_id)

        # R-77 performance (slightly inferior to AIM-120C)
        self.max_speed = 1000.0  # m/s (~Mach 3.5)
        self.boost_acceleration = 350.0  # Good but not as high as AMRAAM
        self.boost_time = 4.5  # Slightly longer burn
        self.max_range = 80000  # 80 km (vs maneuvering target, less than AIM-120)
        self.no_escape_range = 25000  # 25 km (smaller than AMRAAM)
        self.max_g = 35.0  # 35g (slightly less agile)

        # Active radar homing
        self.midcourse_distance = 20000  # 20 km - goes active (later than AMRAAM)
        self.guidance_active = True

        # Warhead
        self.warhead_weight = 22.0  # kg (similar)
        self.lethal_radius = 12.0  # meters (slightly smaller)
