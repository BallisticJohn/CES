"""
F-86F Sabre aircraft model
Based on historical specifications and flight manual data
"""

import numpy as np
from .base import Aircraft
from ..physics.vector import Vector3
from ..physics.quaternion import Quaternion
from ..physics.atmosphere import Atmosphere
from ..aerodynamics.coefficients import F86AeroCoefficients


class F86Sabre(Aircraft):
    """
    F-86F Sabre fighter aircraft

    Based on F-86F-30 variant specifications:
    - Engine: General Electric J47-GE-27 turbojet
    - Maximum thrust: 5,910 lbf (26.3 kN) at sea level
    - Combat weight: ~15,200 lb (6,894 kg)
    - Wing area: 287.9 ft² (26.76 m²)
    - Wing span: 37.1 ft (11.3 m)
    - Maximum speed: 687 mph (1,105 km/h, 596 kts) at sea level
    - Service ceiling: 49,000 ft (14,935 m)
    - Rate of climb: 9,300 ft/min (47.2 m/s)
    - Armament: 6× .50 cal M3 Browning (1,800 rounds total, 300 per gun)
    """

    def __init__(self, position: Vector3 = None, velocity: Vector3 = None,
                 orientation: Quaternion = None):
        super().__init__()

        # Physical specifications (SI units)
        self.mass = 6894.0              # kg (15,200 lb combat weight)
        self.empty_mass = 4967.0        # kg (10,950 lb empty weight)
        self.fuel_mass = 1627.0         # kg (internal fuel)
        self.max_mass = 8936.0          # kg (19,700 lb max takeoff)

        # Geometry
        self.wing_area = 26.76          # m² (287.9 ft²)
        self.wing_span = 11.3           # m (37.1 ft)
        self.wing_chord = 2.37          # m (mean aerodynamic chord)
        self.length = 11.4              # m (37.5 ft)

        # Engine specifications
        self.max_thrust_sealevel = 26300.0   # N (5,910 lbf)
        self.max_thrust = self.max_thrust_sealevel

        # Moment of inertia (based on similar aircraft data)
        # [Ixx (roll), Iyy (pitch), Izz (yaw)] in kg·m²
        # Reduced to realistic values for responsive maneuvering
        self.inertia = np.diag([3500.0, 8000.0, 10000.0])

        # Aerodynamic coefficients
        self.aero_coeffs = F86AeroCoefficients()

        # Performance specifications (for validation)
        self.max_speed_sealevel = Atmosphere.mph_to_mps(687)  # m/s
        self.max_speed_altitude = Atmosphere.mph_to_mps(695)  # m/s at 35,000 ft
        self.service_ceiling = Atmosphere.feet_to_meters(49000)  # m
        self.max_climb_rate = 47.2      # m/s (9,300 ft/min)

        # Weapon system
        self.gun_count = 6
        self.rounds_per_gun = 300
        self.ammo_remaining = self.gun_count * self.rounds_per_gun  # 1,800 rounds
        self.rate_of_fire = 1200        # rounds/min per gun
        self.muzzle_velocity = 884.0    # m/s (.50 BMG)

        # Flight limits
        self.max_load_factor = 7.5      # G's (combat)
        self.never_exceed_speed = Atmosphere.mph_to_mps(710)  # m/s

        # Set initial conditions
        if position is None:
            position = Vector3(0, 0, -Atmosphere.feet_to_meters(20000))  # 20,000 ft
        if velocity is None:
            velocity = Vector3(Atmosphere.mph_to_mps(500), 0, 0)  # 500 mph
        if orientation is None:
            orientation = Quaternion.identity()

        self.set_initial_conditions(
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=Vector3.zero()
        )

    def get_thrust(self, throttle: float, altitude: float, mach: float) -> float:
        """
        Calculate J47-GE-27 engine thrust

        Thrust decreases with altitude and increases slightly with Mach number
        up to about Mach 0.8, then decreases in transonic region.

        Args:
            throttle: Throttle setting (0-1)
            altitude: Altitude (m)
            mach: Mach number

        Returns:
            Thrust in Newtons
        """
        # Base thrust at sea level
        thrust = self.max_thrust_sealevel * throttle

        # Altitude correction (turbojet thrust decreases with density)
        _, _, rho, _ = Atmosphere.get_properties(altitude)
        rho_ratio = rho / Atmosphere.RHO0

        # Turbojet thrust approximately proportional to density ratio
        thrust *= rho_ratio

        # Mach number correction (ram effect)
        # Slight increase up to M=0.8, then decrease
        if mach < 0.8:
            mach_factor = 1.0 + 0.05 * mach  # 5% increase at M=0.8
        else:
            # Inlet losses in transonic
            mach_factor = 1.04 - 0.2 * (mach - 0.8)

        thrust *= max(0.7, mach_factor)  # Don't go below 70% due to compressibility

        return thrust

    def get_corner_velocity(self, altitude: float) -> float:
        """
        Calculate corner velocity (max instantaneous turn rate)

        Args:
            altitude: Altitude (m)

        Returns:
            Corner velocity (m/s)
        """
        _, _, rho, _ = Atmosphere.get_properties(altitude)
        n_max = self.max_load_factor
        W = self.mass * Atmosphere.G

        # V_corner = sqrt(2 * W * n / (rho * S * CL_max))
        CL_max = 1.2  # Approximate max lift coefficient
        V_corner = np.sqrt(2 * W * n_max / (rho * self.wing_area * CL_max))

        return V_corner

    def get_sustained_turn_rate(self, altitude: float, velocity: float) -> float:
        """
        Calculate sustained turn rate at given conditions

        Args:
            altitude: Altitude (m)
            velocity: Airspeed (m/s)

        Returns:
            Turn rate (rad/s)
        """
        # Simplified calculation
        # For sustained turn: Thrust = Drag at load factor n
        # Turn rate = g * sqrt(n² - 1) / V

        _, _, rho, _ = Atmosphere.get_properties(altitude)
        mach = Atmosphere.get_mach_number(velocity, altitude)
        thrust = self.get_thrust(1.0, altitude, mach)

        # Estimate achievable load factor
        q = 0.5 * rho * velocity ** 2
        L_available = q * self.wing_area * 1.0  # CL=1.0 for sustained turn

        n = min(L_available / (self.mass * Atmosphere.G), self.max_load_factor)

        if n > 1.0:
            turn_rate = Atmosphere.G * np.sqrt(n**2 - 1) / velocity
        else:
            turn_rate = 0.0

        return turn_rate

    def get_specific_excess_power(self, altitude: float, velocity: float) -> float:
        """
        Calculate specific excess power (Ps)

        Ps = (T - D) * V / W = rate of energy gain

        Args:
            altitude: Altitude (m)
            velocity: Airspeed (m/s)

        Returns:
            Specific excess power (m/s)
        """
        from ..aerodynamics.forces import AerodynamicForces

        velocity_body = Vector3(velocity, 0, 0)
        alpha, _ = AerodynamicForces.calculate_alpha_beta(velocity_body)
        mach = Atmosphere.get_mach_number(velocity, altitude)

        CL, CD, _ = self.get_aerodynamic_coefficients(alpha, 0.0, mach)

        _, _, rho, _ = Atmosphere.get_properties(altitude)
        q = 0.5 * rho * velocity ** 2

        thrust = self.get_thrust(1.0, altitude, mach)
        drag = q * self.wing_area * CD
        weight = self.mass * Atmosphere.G

        Ps = (thrust - drag) * velocity / weight

        return Ps

    def __str__(self) -> str:
        """String representation"""
        metrics = self.get_performance_metrics()
        return (f"F-86F Sabre\n"
                f"  Position: {self.state.position}\n"
                f"  Altitude: {metrics['altitude_ft']:.0f} ft\n"
                f"  Airspeed: {metrics['airspeed_mph']:.0f} mph "
                f"(Mach {metrics['mach']:.2f})\n"
                f"  Climb rate: {metrics['climb_rate_fpm']:.0f} ft/min\n"
                f"  G-force: {metrics['g_force']:.1f}\n"
                f"  Ammo: {self.ammo_remaining}/{self.gun_count * self.rounds_per_gun}")
