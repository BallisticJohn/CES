"""
Base aircraft class
Defines interface and common functionality for all aircraft
"""

import numpy as np
from abc import ABC, abstractmethod
from ..physics.vector import Vector3
from ..physics.quaternion import Quaternion
from ..physics.atmosphere import Atmosphere
from ..aerodynamics.forces import AerodynamicForces
from ..aerodynamics.coefficients import AeroCoefficients
from ..simulation.integrator import FlightState, StateDerivative


class Aircraft(ABC):
    """
    Base class for all aircraft

    Implements 6-DOF flight dynamics with realistic aerodynamics
    """

    def __init__(self):
        # Physical properties (to be set by subclasses)
        self.mass = 0.0              # kg
        self.wing_area = 0.0         # m²
        self.wing_span = 0.0         # m
        self.wing_chord = 0.0        # m (mean aerodynamic chord)
        self.max_thrust = 0.0        # N

        # Moment of inertia tensor (kg·m²) in body frame
        self.inertia = np.eye(3)

        # Aerodynamic coefficients
        self.aero_coeffs: AeroCoefficients = None

        # Current state
        self.state = FlightState(
            position=Vector3.zero(),
            velocity=Vector3(100, 0, 0),  # Default 100 m/s forward
            orientation=Quaternion.identity(),
            angular_velocity=Vector3.zero(),
            time=0.0
        )

        # Control inputs (normalized -1 to 1)
        self.throttle = 0.8
        self.elevator = 0.0
        self.aileron = 0.0
        self.rudder = 0.0

        # Weapon state
        self.ammo_remaining = 0
        self.is_firing = False

        # Damage state
        self.damage_level = 0.0  # 0.0 = no damage, 1.0 = destroyed
        self.is_destroyed = False

    def set_initial_conditions(self, position: Vector3 = None,
                               velocity: Vector3 = None,
                               orientation: Quaternion = None,
                               angular_velocity: Vector3 = None):
        """Set initial flight conditions"""
        if position is not None:
            self.state.position = position.clone()
        if velocity is not None:
            self.state.velocity = velocity.clone()
        if orientation is not None:
            self.state.orientation = orientation.clone()
        if angular_velocity is not None:
            self.state.angular_velocity = angular_velocity.clone()

    def get_altitude(self) -> float:
        """Get current altitude (m, positive up)"""
        return -self.state.position.z

    def get_airspeed(self) -> float:
        """Get current airspeed (m/s)"""
        return self.state.velocity.magnitude()

    def get_mach_number(self) -> float:
        """Get current Mach number"""
        altitude = self.get_altitude()
        airspeed = self.get_airspeed()
        return Atmosphere.get_mach_number(airspeed, altitude)

    def get_velocity_body(self) -> Vector3:
        """Get velocity in body frame"""
        return self.state.orientation.conjugate().rotate_vector(self.state.velocity)

    @abstractmethod
    def get_thrust(self, throttle: float, altitude: float, mach: float) -> float:
        """
        Calculate engine thrust

        Args:
            throttle: Throttle setting (0-1)
            altitude: Altitude (m)
            mach: Mach number

        Returns:
            Thrust in Newtons
        """
        pass

    def get_aerodynamic_coefficients(self, alpha: float, beta: float,
                                     mach: float) -> tuple[float, float, float]:
        """
        Get aerodynamic coefficients

        Args:
            alpha: Angle of attack (radians)
            beta: Sideslip angle (radians)
            mach: Mach number

        Returns:
            (CL, CD, CY) - lift, drag, side force coefficients
        """
        if self.aero_coeffs is None:
            # Fallback to simple model
            CL = AeroCoefficients.simple_lift_coefficient(alpha)
            CD = AeroCoefficients.simple_drag_coefficient(CL)
            CY = AeroCoefficients.side_force_coefficient(beta)
        else:
            CL = self.aero_coeffs.get_CL(alpha, mach)
            CD = self.aero_coeffs.get_CD(CL, mach)
            CY = AeroCoefficients.side_force_coefficient(beta)

        return CL, CD, CY

    def compute_forces_and_moments(self) -> tuple[Vector3, Vector3]:
        """
        Compute total forces and moments acting on aircraft

        Returns:
            (force, moment) in world frame and body frame respectively
        """
        # Get current flight parameters
        altitude = self.get_altitude()
        velocity_body = self.get_velocity_body()
        V = velocity_body.magnitude()
        mach = self.get_mach_number()

        # Atmospheric properties
        _, _, rho, _ = Atmosphere.get_properties(altitude)

        # Calculate aerodynamic angles
        alpha, beta = AerodynamicForces.calculate_alpha_beta(velocity_body)

        # Get aerodynamic coefficients
        CL, CD, CY = self.get_aerodynamic_coefficients(alpha, beta, mach)

        # Dynamic pressure
        q = AerodynamicForces.calculate_dynamic_pressure(V, rho)

        # Aerodynamic forces in body frame
        F_aero = AerodynamicForces.calculate_lift_drag_forces(
            q, self.wing_area, CL, CD, alpha, beta
        )

        # Add side force
        F_aero.y += q * self.wing_area * CY

        # Engine thrust
        thrust = self.get_thrust(self.throttle, altitude, mach)
        F_thrust = Vector3(thrust, 0, 0)

        # Gravity in body frame
        F_gravity = AerodynamicForces.calculate_gravity_force_body(
            self.mass, self.state.orientation
        )

        # Total force in body frame
        F_body = F_aero + F_thrust + F_gravity

        # Convert to world frame for integration
        F_world = self.state.orientation.rotate_vector(F_body)

        # Moments (simplified - can be expanded)
        # For now, we'll use damping to stabilize
        omega = self.state.angular_velocity

        # Clamp angular velocity to prevent instability
        omega_mag = omega.magnitude()
        max_omega = 5.0  # rad/s max (increased from 3.0 for tighter turns)
        if omega_mag > max_omega:
            omega = omega * (max_omega / omega_mag)

        # REDUCED damping to allow turns
        M_damping = Vector3(
            -0.1 * q * self.wing_area * self.wing_span * omega.x,  # Reduced from 0.5
            -0.2 * q * self.wing_area * self.wing_chord * omega.y,  # Reduced from 1.0
            -0.1 * q * self.wing_area * self.wing_span * omega.z    # Reduced from 0.3
        )

        # VERY HIGH control effectiveness for responsive dogfighting
        # These need to be large enough to generate rapid turns
        M_control = Vector3(
            self.aileron * q * self.wing_area * self.wing_span * 8.0,    # Much higher!
            self.elevator * q * self.wing_area * self.wing_chord * 10.0,  # Much higher!
            self.rudder * q * self.wing_area * self.wing_span * 5.0       # Much higher!
        )

        M_total = M_damping + M_control

        # Clamp total moment to prevent numerical issues
        M_mag = M_total.magnitude()
        max_moment = 1e6  # Newton-meters
        if M_mag > max_moment:
            M_total = M_total * (max_moment / M_mag)

        return F_world, M_total

    def compute_state_derivative(self, state: FlightState, time: float) -> StateDerivative:
        """
        Compute state derivative for integration

        Args:
            state: Current flight state
            time: Current simulation time

        Returns:
            State time derivative
        """
        # Temporarily update state for force calculation
        old_state = self.state
        self.state = state

        # Compute forces and moments
        force, moment = self.compute_forces_and_moments()

        # Restore state
        self.state = old_state

        # Use integrator to compute derivative
        from ..simulation.integrator import RK4Integrator
        return RK4Integrator.compute_state_derivative(
            state, force, moment, self.mass, self.inertia
        )

    def update(self, dt: float):
        """
        Update aircraft state by one time step

        Args:
            dt: Time step (seconds)
        """
        from ..simulation.integrator import RK4Integrator

        # Clamp control inputs to safe ranges
        self.throttle = np.clip(self.throttle, 0.0, 1.0)
        self.elevator = np.clip(self.elevator, -1.0, 1.0)
        self.aileron = np.clip(self.aileron, -1.0, 1.0)
        self.rudder = np.clip(self.rudder, -1.0, 1.0)

        # Integrate one step
        self.state = RK4Integrator.step(
            self.state, dt,
            lambda s, t: self.compute_state_derivative(s, t)
        )

        # Clamp angular velocity after integration
        omega_mag = self.state.angular_velocity.magnitude()
        max_omega = 5.0  # rad/s (increased for tighter turns)
        if omega_mag > max_omega:
            scale = max_omega / omega_mag
            self.state.angular_velocity = self.state.angular_velocity * scale

        # Sanity check for numerical stability
        if not np.isfinite(self.state.position.magnitude()):
            raise RuntimeError("Numerical instability detected in aircraft state")

    def get_performance_metrics(self) -> dict:
        """Get current performance metrics"""
        altitude = self.get_altitude()
        airspeed = self.get_airspeed()
        mach = self.get_mach_number()

        # Climb rate
        climb_rate = -self.state.velocity.z

        # Turn rate
        turn_rate = self.state.angular_velocity.magnitude()

        # G-force
        velocity_body = self.get_velocity_body()
        _, _, rho, _ = Atmosphere.get_properties(altitude)
        alpha, _ = AerodynamicForces.calculate_alpha_beta(velocity_body)
        CL, CD, CY = self.get_aerodynamic_coefficients(alpha, 0, mach)
        q = AerodynamicForces.calculate_dynamic_pressure(airspeed, rho)
        lift_force = q * self.wing_area * CL
        g_force = lift_force / (self.mass * Atmosphere.G)

        return {
            'altitude_m': altitude,
            'altitude_ft': Atmosphere.meters_to_feet(altitude),
            'airspeed_mps': airspeed,
            'airspeed_mph': Atmosphere.mps_to_mph(airspeed),
            'airspeed_knots': Atmosphere.mps_to_knots(airspeed),
            'mach': mach,
            'climb_rate_mps': climb_rate,
            'climb_rate_fpm': climb_rate / 0.3048 * 60,
            'turn_rate_dps': np.degrees(turn_rate),
            'g_force': g_force,
            'alpha_deg': np.degrees(alpha),
        }
