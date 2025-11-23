"""
Tactical geometry calculations
Essential for BFM decision making
"""

import numpy as np
from ..physics.vector import Vector3
from ..physics.quaternion import Quaternion
from typing import Tuple


class TacticalGeometry:
    """Calculate tactical parameters between two aircraft"""

    @staticmethod
    def get_range(pos_self: Vector3, pos_target: Vector3) -> float:
        """
        Calculate slant range to target

        Args:
            pos_self: Own position
            pos_target: Target position

        Returns:
            Range in meters
        """
        return pos_self.distance_to(pos_target)

    @staticmethod
    def get_aspect_angle(pos_self: Vector3, vel_self: Vector3,
                        pos_target: Vector3) -> float:
        """
        Calculate aspect angle: angle between your velocity vector and line to target

        Args:
            pos_self: Own position
            vel_self: Own velocity
            pos_target: Target position

        Returns:
            Aspect angle in radians (0 = nose-on, pi = tail chase)
        """
        to_target = (pos_target - pos_self).normalized()
        vel_norm = vel_self.normalized()

        if vel_norm.magnitude() < 0.1:
            return 0.0

        return vel_norm.angle_to(to_target)

    @staticmethod
    def get_angle_off_tail(pos_self: Vector3, pos_target: Vector3,
                          vel_target: Vector3) -> float:
        """
        Calculate angle-off-tail (AOT): how far off target's six o'clock you are

        Args:
            pos_self: Own position
            pos_target: Target position
            vel_target: Target velocity

        Returns:
            AOT in radians (0 = pure six, pi = head-on)
        """
        to_self = (pos_self - pos_target).normalized()
        target_vel_norm = vel_target.normalized()

        if target_vel_norm.magnitude() < 0.1:
            return 0.0

        # Angle between target's velocity and vector to us
        # Small AOT = we're behind them
        return target_vel_norm.angle_to(to_self)

    @staticmethod
    def get_heading_crossing_angle(vel_self: Vector3, vel_target: Vector3) -> float:
        """
        Calculate heading crossing angle (HCA)

        Args:
            vel_self: Own velocity
            vel_target: Target velocity

        Returns:
            HCA in radians (0 = same direction, pi = opposite)
        """
        v1 = vel_self.normalized()
        v2 = vel_target.normalized()

        if v1.magnitude() < 0.1 or v2.magnitude() < 0.1:
            return 0.0

        return v1.angle_to(v2)

    @staticmethod
    def get_altitude_advantage(pos_self: Vector3, pos_target: Vector3) -> float:
        """
        Calculate altitude advantage (positive = we're higher)

        Args:
            pos_self: Own position (NED frame, Z down)
            pos_target: Target position

        Returns:
            Altitude difference in meters (positive = we're higher)
        """
        return pos_target.z - pos_self.z

    @staticmethod
    def get_closure_rate(pos_self: Vector3, vel_self: Vector3,
                        pos_target: Vector3, vel_target: Vector3) -> float:
        """
        Calculate closure rate (negative = opening)

        Args:
            pos_self: Own position
            vel_self: Own velocity
            pos_target: Target position
            vel_target: Target velocity

        Returns:
            Closure rate in m/s (positive = closing)
        """
        range_vec = pos_target - pos_self
        range_dist = range_vec.magnitude()

        if range_dist < 1.0:
            return 0.0

        range_unit = range_vec / range_dist
        relative_vel = vel_self - vel_target

        # Dot product gives rate of change of range
        # Negative because we want positive = closing
        return -relative_vel.dot(range_unit)

    @staticmethod
    def is_offensive(aot: float, range: float, max_range: float = 1000.0) -> bool:
        """
        Determine if in offensive position

        Args:
            aot: Angle off tail (radians)
            range: Range to target (m)
            max_range: Maximum effective range (m)

        Returns:
            True if offensive (behind target in range)
        """
        # Offensive: small AOT (< 60 deg) and in range
        return aot < np.radians(60) and range < max_range

    @staticmethod
    def is_defensive(aot_bandit: float, range: float, max_range: float = 1500.0) -> bool:
        """
        Determine if defensive (bandit has advantage)

        Args:
            aot_bandit: Bandit's angle off our tail (radians)
            range: Range to bandit (m)
            max_range: Threat range (m)

        Returns:
            True if defensive
        """
        # Defensive: bandit is behind us and close
        return aot_bandit < np.radians(60) and range < max_range

    @staticmethod
    def get_pursuit_type(aspect: float) -> str:
        """
        Determine pursuit type based on aspect

        Args:
            aspect: Aspect angle (radians)

        Returns:
            'pure', 'lead', or 'lag'
        """
        aspect_deg = np.degrees(aspect)

        # Pure pursuit: nose pointing at target (small aspect)
        if aspect_deg < 5:
            return 'pure'
        # Lead pursuit: nose ahead of target
        elif aspect_deg < 85:
            return 'lead'
        # Lag pursuit: nose behind target's flight path
        else:
            return 'lag'

    @staticmethod
    def get_turn_circle_geometry(vel: Vector3, turn_rate: float) -> Tuple[float, Vector3]:
        """
        Calculate turn circle parameters

        Args:
            vel: Velocity vector
            turn_rate: Turn rate (rad/s)

        Returns:
            (radius, center_offset) - radius in meters, offset vector to center
        """
        V = vel.magnitude()

        if turn_rate < 0.001:
            return float('inf'), Vector3.zero()

        # Turn radius = V / omega
        radius = V / turn_rate

        # Center is perpendicular to velocity
        # For level turn, it's 90 degrees to the right
        vel_norm = vel.normalized()

        # Assuming level turn, center is to the right
        # This is simplified - full implementation would use orientation
        center_offset = Vector3(-vel_norm.y, vel_norm.x, 0) * radius

        return radius, center_offset

    @staticmethod
    def predict_intercept_point(pos_self: Vector3, vel_self: Vector3,
                               pos_target: Vector3, vel_target: Vector3,
                               time_step: float = 0.1, max_time: float = 30.0) -> Tuple[Vector3, float]:
        """
        Predict intercept point using simple iterative method

        Args:
            pos_self: Own position
            vel_self: Own velocity
            pos_target: Target position
            vel_target: Target velocity
            time_step: Integration step (s)
            max_time: Maximum prediction time (s)

        Returns:
            (intercept_point, intercept_time) or (target_pos, 0) if no solution
        """
        V_self = vel_self.magnitude()

        if V_self < 1.0:
            return pos_target.clone(), 0.0

        # Iterative prediction
        for t in np.arange(time_step, max_time, time_step):
            # Where target will be
            target_future_pos = pos_target + vel_target * t

            # How far we can travel
            range_we_can_cover = V_self * t

            # Distance to predicted position
            range_to_predicted = pos_self.distance_to(target_future_pos)

            # If we can reach it, that's our intercept
            if abs(range_to_predicted - range_we_can_cover) < V_self * time_step:
                return target_future_pos, t

        # No intercept found, aim at current position
        return pos_target.clone(), 0.0

    @staticmethod
    def compute_lead_angle(pos_self: Vector3, vel_self: Vector3,
                          pos_target: Vector3, vel_target: Vector3) -> Vector3:
        """
        Compute lead vector for guns solution

        Args:
            pos_self: Own position
            vel_self: Own velocity
            pos_target: Target position
            vel_target: Target velocity

        Returns:
            Lead point vector (direction to aim)
        """
        intercept_point, _ = TacticalGeometry.predict_intercept_point(
            pos_self, vel_self, pos_target, vel_target
        )

        return (intercept_point - pos_self).normalized()
