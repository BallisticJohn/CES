"""
Pilot AI controller
Translates BFM decisions into aircraft control inputs
"""

import numpy as np
from typing import TYPE_CHECKING, Optional
from ..physics.vector import Vector3
from ..physics.quaternion import Quaternion
from ..physics.atmosphere import Atmosphere
from .tactics import TacticalGeometry
from .maneuvers import BFMManeuvers

if TYPE_CHECKING:
    from ..aircraft.base import Aircraft


class PilotAI:
    """
    AI pilot controller

    Implements:
    - Tactical assessment
    - Maneuver selection
    - Control input computation
    - Energy management
    """

    def __init__(self, aircraft: 'Aircraft', aggression: float = 0.7,
                 skill_level: float = 0.8):
        """
        Initialize pilot AI

        Args:
            aircraft: Aircraft to control
            aggression: How aggressive (0-1), affects energy management
            skill_level: Pilot skill (0-1), affects precision and reaction
        """
        self.aircraft = aircraft
        self.aggression = np.clip(aggression, 0.0, 1.0)
        self.skill_level = np.clip(skill_level, 0.0, 1.0)

        # Tactical state
        self.current_maneuver = 'none'
        self.maneuver_start_time = 0.0
        self.target_aircraft: Optional['Aircraft'] = None

        # Control gains (tuned for responsiveness and stability)
        self.heading_gain = 0.5 * skill_level
        self.pitch_gain = 0.3 * skill_level
        self.altitude_gain = 0.1
        self.speed_gain = 0.1

        # Energy management preferences
        self.min_speed = Atmosphere.mph_to_mps(250)  # Don't go below 250 mph
        self.max_speed = Atmosphere.mph_to_mps(650)  # Don't exceed 650 mph
        self.preferred_speed = Atmosphere.mph_to_mps(450)  # Corner velocity region

    def set_target(self, target: 'Aircraft'):
        """Set target aircraft"""
        self.target_aircraft = target

    def update(self, dt: float):
        """
        Update pilot control inputs

        Args:
            dt: Time step (s)
        """
        if self.target_aircraft is None:
            # No target, maintain altitude and speed
            self._maintain_level_flight()
            return

        # Get tactical situation
        tactical = self._assess_situation()

        # Select maneuver based on situation
        self._select_maneuver(tactical)

        # Execute maneuver
        self._execute_maneuver(tactical)

        # Manage energy (throttle control)
        self._manage_energy(tactical)

    def _assess_situation(self) -> dict:
        """Assess tactical situation"""
        pos_self = self.aircraft.state.position
        vel_self = self.aircraft.state.velocity
        pos_target = self.target_aircraft.state.position
        vel_target = self.target_aircraft.state.velocity

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
        bandit_aot = TacticalGeometry.get_angle_off_tail(
            pos_target, pos_self, vel_self
        )

        # Our current state
        altitude = self.aircraft.get_altitude()
        airspeed = self.aircraft.get_airspeed()
        mach = self.aircraft.get_mach_number()

        # Specific excess power
        ps = self.aircraft.get_specific_excess_power(altitude, airspeed)

        return {
            'range': range_to_target,
            'aot': aot,  # Our angle off target's tail
            'aspect': aspect,
            'closure': closure,
            'alt_adv': alt_adv,
            'hca': hca,
            'bandit_aot': bandit_aot,  # Bandit's angle off our tail
            'altitude': altitude,
            'airspeed': airspeed,
            'mach': mach,
            'ps': ps,
            'offensive': TacticalGeometry.is_offensive(aot, range_to_target),
            'defensive': TacticalGeometry.is_defensive(bandit_aot, range_to_target),
        }

    def _select_maneuver(self, tactical: dict):
        """Select appropriate maneuver"""
        self.current_maneuver = BFMManeuvers.select_maneuver(
            self.aircraft.state.position,
            self.aircraft.state.velocity,
            self.target_aircraft.state.position,
            self.target_aircraft.state.velocity,
            tactical['altitude']
        )

    def _execute_maneuver(self, tactical: dict):
        """Execute selected maneuver by setting control inputs"""
        pos_self = self.aircraft.state.position
        vel_self = self.aircraft.state.velocity
        pos_target = self.target_aircraft.state.position
        vel_target = self.target_aircraft.state.velocity

        # Get desired heading and pitch from maneuver
        if self.current_maneuver == 'pure_pursuit':
            desired_dir, desired_pitch = BFMManeuvers.pure_pursuit(
                pos_self, vel_self, pos_target
            )
        elif self.current_maneuver == 'lead_pursuit':
            desired_dir, desired_pitch = BFMManeuvers.lead_pursuit(
                pos_self, vel_self, pos_target, vel_target
            )
        elif self.current_maneuver == 'lag_pursuit':
            desired_dir, desired_pitch = BFMManeuvers.lag_pursuit(
                pos_self, vel_self, pos_target, vel_target
            )
        elif self.current_maneuver == 'high_yo_yo':
            desired_dir, desired_pitch = BFMManeuvers.high_yo_yo(
                pos_self, vel_self, pos_target, vel_target
            )
        elif self.current_maneuver == 'low_yo_yo':
            desired_dir, desired_pitch = BFMManeuvers.low_yo_yo(
                pos_self, vel_self, pos_target, vel_target
            )
        elif self.current_maneuver == 'defensive_break':
            desired_dir, desired_pitch, desired_roll = BFMManeuvers.defensive_break(
                pos_self, vel_self, pos_target, vel_target
            )
            # Max performance break
            self.aircraft.aileron = np.sign(desired_roll) * 1.0
        else:
            # Default to pure pursuit
            desired_dir, desired_pitch = BFMManeuvers.pure_pursuit(
                pos_self, vel_self, pos_target
            )

        # Convert desired direction to control inputs
        self._set_controls_for_direction(desired_dir, desired_pitch)

    def _set_controls_for_direction(self, desired_dir: Vector3, desired_pitch: float):
        """
        Set elevator and aileron to point aircraft in desired direction

        Args:
            desired_dir: Desired direction vector (world frame)
            desired_pitch: Desired pitch angle (radians)
        """
        # Get current aircraft orientation
        orientation = self.aircraft.state.orientation

        # Convert desired direction to body frame
        desired_dir_body = orientation.conjugate().rotate_vector(desired_dir)

        # Calculate errors in body frame
        # X = forward, Y = right, Z = down

        # Heading error: atan2(Y, X) - positive right
        heading_error = np.arctan2(desired_dir_body.y, desired_dir_body.x)

        # Pitch error: arctan2(-Z, X) - positive up
        current_pitch = np.arctan2(-desired_dir_body.z,
                                   np.sqrt(desired_dir_body.x**2 + desired_dir_body.y**2))
        pitch_error = desired_pitch - current_pitch

        # Set controls with gains and rate limiting
        # Aileron for heading (roll to turn)
        desired_aileron = heading_error * self.heading_gain
        # Rate limit to prevent violent control inputs
        max_rate = 0.5  # Max change per update
        current_aileron = self.aircraft.aileron
        delta = np.clip(desired_aileron - current_aileron, -max_rate, max_rate)
        self.aircraft.aileron = np.clip(current_aileron + delta, -0.8, 0.8)

        # Elevator for pitch
        desired_elevator = pitch_error * self.pitch_gain
        current_elevator = self.aircraft.elevator
        delta = np.clip(desired_elevator - current_elevator, -max_rate, max_rate)
        self.aircraft.elevator = np.clip(current_elevator + delta, -0.8, 0.8)

        # Rudder for coordination (simplified, very gentle)
        self.aircraft.rudder = np.clip(
            -heading_error * 0.05,
            -0.3, 0.3
        )

    def _manage_energy(self, tactical: dict):
        """
        Manage throttle for energy state

        Args:
            tactical: Tactical situation dictionary
        """
        airspeed = tactical['airspeed']
        ps = tactical['ps']

        # Energy management strategy
        if self.current_maneuver == 'defensive_break':
            # Max power in defense
            self.aircraft.throttle = 1.0

        elif self.current_maneuver == 'low_yo_yo':
            # Dive = less thrust needed
            self.aircraft.throttle = 0.7

        elif self.current_maneuver == 'high_yo_yo':
            # Climb = max thrust
            self.aircraft.throttle = 1.0

        elif tactical['offensive']:
            # Offensive: maintain corner velocity
            if airspeed < self.preferred_speed:
                # Too slow, add power
                self.aircraft.throttle = 0.9 + self.aggression * 0.1
            elif airspeed > self.preferred_speed * 1.2:
                # Too fast, reduce
                self.aircraft.throttle = 0.6
            else:
                # Good speed, maintain
                self.aircraft.throttle = 0.85

        else:
            # Neutral/extending: manage for Ps
            if ps < 0:
                # Losing energy, add power
                self.aircraft.throttle = 0.95
            elif airspeed < self.min_speed:
                # Dangerously slow
                self.aircraft.throttle = 1.0
            elif airspeed > self.max_speed:
                # Too fast
                self.aircraft.throttle = 0.5
            else:
                # Good energy state
                self.aircraft.throttle = 0.8 + self.aggression * 0.1

        # Clamp throttle
        self.aircraft.throttle = np.clip(self.aircraft.throttle, 0.0, 1.0)

    def _maintain_level_flight(self):
        """Maintain level flight when no target"""
        # Maintain altitude
        altitude = self.aircraft.get_altitude()
        target_altitude = Atmosphere.feet_to_meters(25000)  # Cruise at 25k ft

        altitude_error = target_altitude - altitude

        # Pitch to correct altitude
        desired_pitch = np.clip(altitude_error * 0.0001, -0.2, 0.2)

        self.aircraft.elevator = desired_pitch * 2.0

        # Wings level
        self.aircraft.aileron = 0.0
        self.aircraft.rudder = 0.0

        # Maintain cruise speed
        airspeed = self.aircraft.get_airspeed()
        if airspeed < self.preferred_speed:
            self.aircraft.throttle = 0.85
        else:
            self.aircraft.throttle = 0.75

    def get_status_string(self, tactical: dict = None) -> str:
        """Get human-readable status string"""
        if tactical is None and self.target_aircraft is not None:
            tactical = self._assess_situation()

        if tactical is None:
            return f"Pilot: No target, maintaining {Atmosphere.meters_to_feet(self.aircraft.get_altitude()):.0f}ft"

        status = f"Maneuver: {self.current_maneuver.upper().replace('_', ' ')}\n"
        status += f"  Range: {tactical['range']/1000:.2f} km\n"
        status += f"  AOT: {np.degrees(tactical['aot']):.0f}°\n"
        status += f"  Aspect: {np.degrees(tactical['aspect']):.0f}°\n"
        status += f"  Closure: {tactical['closure']:.0f} m/s\n"
        status += f"  Position: {'OFFENSIVE' if tactical['offensive'] else 'DEFENSIVE' if tactical['defensive'] else 'NEUTRAL'}\n"
        status += f"  Ps: {tactical['ps']:.1f} m/s\n"

        return status
