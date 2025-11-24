"""
Basic Fighter Maneuvers (BFM) implementation
Classic pursuit curves and energy management
"""

import numpy as np
from ..physics.vector import Vector3
from ..physics.quaternion import Quaternion
from ..physics.atmosphere import Atmosphere
from .tactics import TacticalGeometry
from typing import Tuple


class BFMManeuvers:
    """
    Implementation of basic fighter maneuvers

    References:
    - "Fighter Combat: Tactics and Maneuvering" by Robert Shaw
    - "The Art of the Kill" by Pete Bonanni (USAF)
    """

    @staticmethod
    def pure_pursuit(pos_self: Vector3, vel_self: Vector3,
                    pos_target: Vector3) -> Tuple[Vector3, float]:
        """
        Pure pursuit: Point nose directly at target
        Simple but bleeds energy and can overshoot

        Args:
            pos_self: Own position
            vel_self: Own velocity
            pos_target: Target position

        Returns:
            (desired_heading, desired_pitch) - unit direction vector and pitch angle
        """
        # Vector to target
        to_target = (pos_target - pos_self).normalized()

        # Pitch angle
        pitch = -np.arcsin(np.clip(to_target.z, -1.0, 1.0))

        return to_target, pitch

    @staticmethod
    def lead_pursuit(pos_self: Vector3, vel_self: Vector3,
                    pos_target: Vector3, vel_target: Vector3,
                    lead_factor: float = 1.5) -> Tuple[Vector3, float]:
        """
        Lead pursuit: Point ahead of target's flight path
        Used to decrease range and get gun solution

        Args:
            pos_self: Own position
            vel_self: Own velocity
            pos_target: Target position
            vel_target: Target velocity
            lead_factor: How much to lead (1.0 = pure, >1.0 = lead)

        Returns:
            (desired_heading, desired_pitch)
        """
        # Predict where target will be
        intercept_point, dt = TacticalGeometry.predict_intercept_point(
            pos_self, vel_self, pos_target, vel_target
        )

        # Lead even more for aggressive closure
        lead_point = pos_target + vel_target * (dt * lead_factor)

        to_lead = (lead_point - pos_self).normalized()
        pitch = -np.arcsin(np.clip(to_lead.z, -1.0, 1.0))

        return to_lead, pitch

    @staticmethod
    def lag_pursuit(pos_self: Vector3, vel_self: Vector3,
                   pos_target: Vector3, vel_target: Vector3,
                   lag_angle: float = 0.3) -> Tuple[Vector3, float]:
        """
        Lag pursuit: Point behind target's flight path
        Used to control closure rate and preserve energy

        Args:
            pos_self: Own position
            vel_self: Own velocity
            pos_target: Target position
            vel_target: Target velocity
            lag_angle: How much to lag in radians

        Returns:
            (desired_heading, desired_pitch)
        """
        # Get vector to target
        to_target = (pos_target - pos_self).normalized()

        # Target's velocity direction
        target_vel_norm = vel_target.normalized()

        # Lag by rotating our aim point behind target's flight path
        # Simple approach: blend between target position and behind target
        behind_target = pos_target - target_vel_norm * 500  # 500m behind

        # Weighted blend
        lag_point = pos_target + (behind_target - pos_target) * np.sin(lag_angle)

        to_lag = (lag_point - pos_self).normalized()
        pitch = -np.arcsin(np.clip(to_lag.z, -1.0, 1.0))

        return to_lag, pitch

    @staticmethod
    def high_yo_yo(pos_self: Vector3, vel_self: Vector3,
                  pos_target: Vector3, vel_target: Vector3,
                  altitude_gain: float = 500.0) -> Tuple[Vector3, float]:
        """
        High yo-yo: Pull up and over to decrease turn radius
        Trades energy for position

        Args:
            pos_self: Own position
            vel_self: Own velocity
            pos_target: Target position
            vel_target: Target velocity
            altitude_gain: Desired altitude increase (m)

        Returns:
            (desired_heading, desired_pitch)
        """
        # Pull up to gain altitude
        current_alt = -pos_self.z
        target_alt = -pos_target.z

        # If we need altitude, pull up
        if target_alt - current_alt < altitude_gain:
            # Pull up while maintaining direction toward target (in horizontal)
            to_target_horiz = pos_target - pos_self
            to_target_horiz.z = 0  # Ignore vertical
            to_target_horiz = to_target_horiz.normalized()

            # Add vertical component for climb
            pitch = np.radians(30)  # 30 degree climb

            return to_target_horiz, pitch
        else:
            # Roll back down toward target
            return BFMManeuvers.lead_pursuit(pos_self, vel_self, pos_target, vel_target)

    @staticmethod
    def low_yo_yo(pos_self: Vector3, vel_self: Vector3,
                 pos_target: Vector3, vel_target: Vector3) -> Tuple[Vector3, float]:
        """
        Low yo-yo: Dive to increase speed and tighten turn
        Trades altitude for energy and position

        Args:
            pos_self: Own position
            vel_self: Own velocity
            pos_target: Target position
            vel_target: Target velocity

        Returns:
            (desired_heading, desired_pitch)
        """
        # Dive toward target while maintaining pursuit
        to_target = (pos_target - pos_self).normalized()

        # Aggressive dive angle
        pitch = np.radians(-20)  # 20 degree dive

        return to_target, pitch

    @staticmethod
    def defensive_break(pos_self: Vector3, vel_self: Vector3,
                       pos_bandit: Vector3, vel_bandit: Vector3,
                       break_direction: str = 'into') -> Tuple[Vector3, float, float]:
        """
        Defensive break turn: Hard turn to defeat attack

        Args:
            pos_self: Own position
            vel_self: Own velocity
            pos_bandit: Bandit position
            vel_bandit: Bandit velocity
            break_direction: 'into' or 'away' from bandit

        Returns:
            (desired_heading, desired_pitch, desired_roll)
        """
        to_bandit = (pos_bandit - pos_self).normalized()

        if break_direction == 'into':
            # Break INTO the bandit (cross his nose)
            # This forces him to pull harder and can cause overshoot
            desired_heading = to_bandit
            roll = np.radians(90)  # Max roll
        else:
            # Break AWAY (extend)
            desired_heading = (pos_self - pos_bandit).normalized()
            roll = np.radians(90)

        # Level turn initially
        pitch = 0.0

        return desired_heading, pitch, roll

    @staticmethod
    def vertical_scissors(pos_self: Vector3, vel_self: Vector3,
                         pos_target: Vector3, vel_target: Vector3,
                         phase: str = 'up') -> Tuple[Vector3, float]:
        """
        Vertical scissors: Rolling vertical maneuver to force overshoot

        Args:
            pos_self: Own position
            vel_self: Own velocity
            pos_target: Target position
            vel_target: Target velocity
            phase: 'up' or 'down'

        Returns:
            (desired_heading, desired_pitch)
        """
        V_self = vel_self.magnitude()

        if phase == 'up':
            # Pull up hard
            to_target = (pos_target - pos_self).normalized()
            pitch = np.radians(45)  # 45 degree climb
            return to_target, pitch
        else:
            # Roll and pull down
            to_target = (pos_target - pos_self).normalized()
            pitch = np.radians(-45)  # 45 degree dive
            return to_target, pitch

    @staticmethod
    def energy_sustaining_turn(airspeed: float, altitude: float,
                              aircraft_type: str = 'f86') -> float:
        """
        Calculate turn rate that sustains energy (Ps = 0)

        Args:
            airspeed: Current airspeed (m/s)
            altitude: Current altitude (m)
            aircraft_type: Aircraft type for performance

        Returns:
            Recommended turn rate (rad/s)
        """
        # This is simplified - real implementation would use aircraft Ps data
        # At corner velocity, max sustained turn

        # Rough estimate based on historical data
        if aircraft_type == 'f86':
            corner_speed = Atmosphere.mph_to_mps(450)  # ~450 mph corner
        else:  # mig15
            corner_speed = Atmosphere.mph_to_mps(420)

        # Closer to corner speed = tighter turns possible
        speed_ratio = airspeed / corner_speed

        if speed_ratio > 1.2:
            # Too fast, gentle turn
            return np.radians(5)
        elif speed_ratio > 0.8:
            # Near corner, max turn
            return np.radians(12)
        else:
            # Too slow, gentle turn to avoid departure
            return np.radians(6)

    @staticmethod
    def select_maneuver(pos_self: Vector3, vel_self: Vector3,
                       pos_target: Vector3, vel_target: Vector3,
                       altitude_self: float) -> str:
        """
        Select appropriate BFM based on tactical situation

        Args:
            pos_self: Own position
            vel_self: Own velocity
            pos_target: Target position
            vel_target: Target velocity
            altitude_self: Current altitude

        Returns:
            Maneuver name
        """
        # Calculate tactical parameters
        range_to_target = TacticalGeometry.get_range(pos_self, pos_target)
        aot = TacticalGeometry.get_angle_off_tail(pos_self, pos_target, vel_target)
        aspect = TacticalGeometry.get_aspect_angle(pos_self, vel_self, pos_target)
        closure = TacticalGeometry.get_closure_rate(
            pos_self, vel_self, pos_target, vel_target
        )
        alt_adv = TacticalGeometry.get_altitude_advantage(pos_self, pos_target)
        hca = TacticalGeometry.get_heading_crossing_angle(vel_self, vel_target)

        # Bandit's angle off our tail (are we defensive?)
        bandit_aot = TacticalGeometry.get_angle_off_tail(pos_target, pos_self, vel_self)

        # CRITICAL: Check if we just passed and are opening
        # High aspect + negative closure = we passed each other
        if aspect > np.radians(120) and closure < -50 and range_to_target < 3000:
            # We just passed! Need to reverse
            if alt_adv > 300:
                # We're higher, use high yo-yo to reverse
                return 'high_yo_yo'
            else:
                # Execute hard reversal turn
                return 'defensive_break'  # Use break turn mechanics to reverse hard

        # OPENING: If we're separating fast at medium range, turn back
        if closure < -100 and range_to_target > 1000 and range_to_target < 5000:
            # Separating, need to turn around
            return 'pure_pursuit'  # Turn hard toward target

        # Decision tree for maneuver selection

        # OFFENSIVE (behind target)
        if TacticalGeometry.is_offensive(aot, range_to_target):
            if range_to_target < 500 and closure > 50:
                # Too much closure, about to overshoot
                return 'lag_pursuit'
            elif range_to_target < 300:
                # Gun range, lead for solution
                return 'lead_pursuit'
            elif range_to_target > 1000 and alt_adv > 200:
                # Far away, use altitude
                return 'low_yo_yo'
            else:
                # Standard attack
                return 'lead_pursuit'

        # DEFENSIVE (target behind us)
        if TacticalGeometry.is_defensive(bandit_aot, range_to_target):
            if range_to_target < 800:
                # Close defensive, break turn
                return 'defensive_break'
            else:
                # Some separation, vertical maneuver
                if alt_adv < -500:
                    return 'low_yo_yo'
                else:
                    return 'high_yo_yo'

        # NEUTRAL (angles fight)
        if range_to_target < 2000:
            # Close neutral, turn fight
            if vel_self.magnitude() < Atmosphere.mph_to_mps(400):
                # Slow, need energy
                return 'low_yo_yo'
            else:
                # Good speed, pure turn
                return 'pure_pursuit'
        else:
            # Extending, turn back
            return 'pure_pursuit'
