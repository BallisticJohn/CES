"""
Aerodynamic force calculations
Converts coefficients and flight conditions into forces and moments
"""

import numpy as np
from ..physics.vector import Vector3
from ..physics.atmosphere import Atmosphere


class AerodynamicForces:
    """Calculate aerodynamic forces and moments"""

    @staticmethod
    def calculate_alpha_beta(velocity_body: Vector3) -> tuple[float, float]:
        """
        Calculate angle of attack and sideslip angle from body-frame velocity

        Args:
            velocity_body: Velocity vector in body frame (m/s)

        Returns:
            (alpha, beta) in radians
            alpha: angle of attack
            beta: sideslip angle
        """
        vx, vy, vz = velocity_body.x, velocity_body.y, velocity_body.z
        V = velocity_body.magnitude()

        if V < 1e-3:
            return 0.0, 0.0

        # Angle of attack (rotation about Y axis)
        alpha = np.arctan2(vz, vx)

        # Sideslip angle (rotation about Z axis)
        beta = np.arcsin(np.clip(vy / V, -1.0, 1.0))

        return alpha, beta

    @staticmethod
    def calculate_dynamic_pressure(velocity: float, density: float) -> float:
        """
        Calculate dynamic pressure

        Args:
            velocity: Airspeed (m/s)
            density: Air density (kg/m³)

        Returns:
            Dynamic pressure q (Pa)
        """
        return 0.5 * density * velocity ** 2

    @staticmethod
    def body_to_wind_matrix(alpha: float, beta: float) -> np.ndarray:
        """
        Rotation matrix from body frame to wind frame

        Args:
            alpha: Angle of attack (radians)
            beta: Sideslip angle (radians)

        Returns:
            3x3 rotation matrix
        """
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        cb = np.cos(beta)
        sb = np.sin(beta)

        return np.array([
            [ca * cb, -ca * sb, -sa],
            [sb, cb, 0],
            [sa * cb, -sa * sb, ca]
        ])

    @staticmethod
    def wind_to_body_matrix(alpha: float, beta: float) -> np.ndarray:
        """
        Rotation matrix from wind frame to body frame

        Args:
            alpha: Angle of attack (radians)
            beta: Sideslip angle (radians)

        Returns:
            3x3 rotation matrix
        """
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        cb = np.cos(beta)
        sb = np.sin(beta)

        return np.array([
            [ca * cb, sb, sa * cb],
            [-ca * sb, cb, -sa * sb],
            [-sa, 0, ca]
        ])

    @staticmethod
    def calculate_lift_drag_forces(q: float, S: float, CL: float, CD: float,
                                   alpha: float, beta: float) -> Vector3:
        """
        Calculate aerodynamic forces in body frame from lift and drag coefficients

        Args:
            q: Dynamic pressure (Pa)
            S: Reference wing area (m²)
            CL: Lift coefficient
            CD: Drag coefficient
            alpha: Angle of attack (radians)
            beta: Sideslip angle (radians)

        Returns:
            Force vector in body frame (N)
        """
        # Forces in wind frame
        # Wind frame: X-drag (opposite to velocity), Y-sideforce, Z-lift (downward positive)
        drag = q * S * CD
        lift = q * S * CL

        # Wind frame forces (Z-down convention)
        F_wind = np.array([-drag, 0, -lift])

        # Convert to body frame
        R_wind_to_body = AerodynamicForces.wind_to_body_matrix(alpha, beta)
        F_body = R_wind_to_body @ F_wind

        return Vector3(F_body[0], F_body[1], F_body[2])

    @staticmethod
    def calculate_total_forces(velocity_body: Vector3, altitude: float,
                              wing_area: float, mass: float,
                              CL: float, CD: float, CY: float,
                              thrust: float) -> Vector3:
        """
        Calculate total forces acting on aircraft in body frame

        Args:
            velocity_body: Velocity in body frame (m/s)
            altitude: Altitude (m)
            wing_area: Reference wing area (m²)
            mass: Aircraft mass (kg)
            CL: Lift coefficient
            CD: Drag coefficient
            CY: Side force coefficient
            thrust: Engine thrust (N)

        Returns:
            Total force vector in body frame (N)
        """
        # Atmospheric properties
        _, _, rho, _ = Atmosphere.get_properties(altitude)

        # Flight parameters
        V = velocity_body.magnitude()
        alpha, beta = AerodynamicForces.calculate_alpha_beta(velocity_body)
        q = AerodynamicForces.calculate_dynamic_pressure(V, rho)

        # Aerodynamic forces
        F_aero = AerodynamicForces.calculate_lift_drag_forces(
            q, wing_area, CL, CD, alpha, beta
        )

        # Side force
        F_side = q * wing_area * CY
        F_aero.y += F_side

        # Thrust (along body X-axis)
        F_thrust = Vector3(thrust, 0, 0)

        # Total aerodynamic + thrust
        F_total = F_aero + F_thrust

        return F_total

    @staticmethod
    def calculate_moments(q: float, S: float, span: float, chord: float,
                         Cl: float, Cm: float, Cn: float) -> Vector3:
        """
        Calculate aerodynamic moments in body frame

        Args:
            q: Dynamic pressure (Pa)
            S: Reference wing area (m²)
            span: Wing span (m)
            chord: Mean aerodynamic chord (m)
            Cl: Rolling moment coefficient
            Cm: Pitching moment coefficient
            Cn: Yawing moment coefficient

        Returns:
            Moment vector in body frame (N·m)
            (L, M, N) = (roll, pitch, yaw)
        """
        L = q * S * span * Cl   # Rolling moment
        M = q * S * chord * Cm  # Pitching moment
        N = q * S * span * Cn   # Yawing moment

        return Vector3(L, M, N)

    @staticmethod
    def calculate_gravity_force_body(mass: float, orientation_quat) -> Vector3:
        """
        Calculate gravity force in body frame

        Args:
            mass: Aircraft mass (kg)
            orientation_quat: Aircraft orientation quaternion

        Returns:
            Gravity force in body frame (N)
        """
        # Gravity in world frame (NED: down is positive)
        g_world = Vector3(0, 0, mass * Atmosphere.G)

        # Rotate to body frame
        g_body = orientation_quat.conjugate().rotate_vector(g_world)

        return g_body
