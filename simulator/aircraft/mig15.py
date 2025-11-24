"""
MiG-15bis aircraft model
Based on historical specifications and declassified data
"""

import numpy as np
from .base import Aircraft
from ..physics.vector import Vector3
from ..physics.quaternion import Quaternion
from ..physics.atmosphere import Atmosphere
from ..aerodynamics.coefficients import MiG15AeroCoefficients


class MiG15(Aircraft):
    """
    MiG-15bis fighter aircraft

    Based on MiG-15bis specifications:
    - Engine: Klimov VK-1 turbojet (license-built Rolls-Royce Nene)
    - Maximum thrust: 5,952 lbf (26.5 kN) at sea level
    - Normal loaded weight: ~11,177 lb (5,070 kg)
    - Wing area: 221.7 ft² (20.6 m²)
    - Wing span: 33.1 ft (10.1 m)
    - Maximum speed: 668 mph (1,075 km/h, 580 kts) at sea level
    - Service ceiling: 50,850 ft (15,500 m)
    - Rate of climb: 10,100 ft/min (51.3 m/s)
    - Armament: 1× 37mm N-37 (40 rounds), 2× 23mm NR-23 (160 rounds total)
    """

    def __init__(self, position: Vector3 = None, velocity: Vector3 = None,
                 orientation: Quaternion = None):
        super().__init__()

        # Physical specifications (SI units)
        self.mass = 5070.0              # kg (11,177 lb normal loaded)
        self.empty_mass = 3584.0        # kg (7,900 lb)
        self.fuel_mass = 1286.0         # kg (internal fuel)
        self.max_mass = 6106.0          # kg (13,460 lb max takeoff)

        # Geometry
        self.wing_area = 20.6           # m² (221.7 ft²)
        self.wing_span = 10.08          # m (33.1 ft)
        self.wing_chord = 2.04          # m (mean aerodynamic chord)
        self.length = 10.1              # m (33.2 ft)

        # Engine specifications
        self.max_thrust_sealevel = 26500.0   # N (5,952 lbf)
        self.max_thrust = self.max_thrust_sealevel

        # Moment of inertia (lighter and smaller than F-86)
        # [Ixx (roll), Iyy (pitch), Izz (yaw)] in kg·m²
        # Reduced to realistic values for nimble maneuvering
        self.inertia = np.diag([2800.0, 6500.0, 8000.0])

        # Aerodynamic coefficients
        self.aero_coeffs = MiG15AeroCoefficients()

        # Performance specifications (for validation)
        self.max_speed_sealevel = Atmosphere.mph_to_mps(668)  # m/s
        self.max_speed_altitude = Atmosphere.mph_to_mps(684)  # m/s at 9,800 ft
        self.service_ceiling = Atmosphere.feet_to_meters(50850)  # m
        self.max_climb_rate = 51.3      # m/s (10,100 ft/min)

        # Weapon system
        # 1× N-37 37mm cannon
        self.cannon_37mm_rounds = 40
        self.cannon_37mm_muzzle_velocity = 690.0  # m/s
        self.cannon_37mm_rate_of_fire = 400  # rounds/min

        # 2× NR-23 23mm cannons
        self.cannon_23mm_count = 2
        self.cannon_23mm_rounds_per_gun = 80
        self.cannon_23mm_muzzle_velocity = 690.0  # m/s
        self.cannon_23mm_rate_of_fire = 650  # rounds/min per gun

        self.ammo_remaining = (self.cannon_37mm_rounds +
                              self.cannon_23mm_count * self.cannon_23mm_rounds_per_gun)  # 200 rounds total

        # Flight limits
        self.max_load_factor = 8.0      # G's (combat, lighter airframe)
        self.never_exceed_speed = Atmosphere.mph_to_mps(690)  # m/s

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
        Calculate VK-1 engine thrust

        The VK-1 (license-built Rolls-Royce Nene) had better high-altitude
        performance than the J47 but similar sea-level thrust.

        Args:
            throttle: Throttle setting (0-1)
            altitude: Altitude (m)
            mach: Mach number

        Returns:
            Thrust in Newtons
        """
        # Base thrust at sea level
        thrust = self.max_thrust_sealevel * throttle

        # Altitude correction
        # VK-1 had slightly better altitude performance than J47
        _, _, rho, _ = Atmosphere.get_properties(altitude)
        rho_ratio = rho / Atmosphere.RHO0

        # Centrifugal engine maintains thrust better at altitude
        altitude_factor = rho_ratio ** 0.85  # Less degradation than 1.0 power

        thrust *= altitude_factor

        # Mach number correction
        if mach < 0.75:
            mach_factor = 1.0 + 0.06 * mach  # Slightly better ram effect
        else:
            # More inlet losses in transonic (swept wing, different inlet design)
            mach_factor = 1.045 - 0.25 * (mach - 0.75)

        thrust *= max(0.65, mach_factor)

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
        CL_max = 1.1  # Slightly lower than F-86 (earlier stall)
        V_corner = np.sqrt(2 * W * n_max / (rho * self.wing_area * CL_max))

        return V_corner

    def get_sustained_turn_rate(self, altitude: float, velocity: float) -> float:
        """
        Calculate sustained turn rate at given conditions

        MiG-15 had better thrust/weight ratio, giving better sustained turn
        performance, especially at altitude.

        Args:
            altitude: Altitude (m)
            velocity: Airspeed (m/s)

        Returns:
            Turn rate (rad/s)
        """
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

        MiG-15 generally had better Ps at high altitude due to:
        - Better thrust/weight ratio
        - Better high-altitude engine performance

        Args:
            altitude: Altitude (m)
            velocity: Airspeed (m/s)

        Returns:
            Specific excess power (m/s)
        """
        # This is a simplified calculation
        # Would need full integration with aerodynamic model for accuracy
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

    def get_advantage_envelope(self, vs_aircraft: str = "F86") -> dict:
        """
        Get performance advantages/disadvantages vs specified aircraft

        Returns dict of comparative advantages
        """
        advantages = {
            'climb_rate': True,           # Better climb
            'high_altitude': True,        # Better ceiling
            'thrust_to_weight': True,     # Better T/W
            'turn_performance_high': True,  # Better sustained turn at altitude
            'turn_performance_low': False,  # Worse at low altitude/high speed
            'transonic': False,           # Worse transonic handling
            'roll_rate': False,           # Slower roll rate
            'dive_speed': False,          # Lower dive speed limit
            'firepower_weight': True,     # Heavier guns (but less ammo)
            'firepower_accuracy': False,  # Harder to aim (different velocities)
        }
        return advantages

    def __str__(self) -> str:
        """String representation"""
        metrics = self.get_performance_metrics()
        return (f"MiG-15bis\n"
                f"  Position: {self.state.position}\n"
                f"  Altitude: {metrics['altitude_ft']:.0f} ft\n"
                f"  Airspeed: {metrics['airspeed_mph']:.0f} mph "
                f"(Mach {metrics['mach']:.2f})\n"
                f"  Climb rate: {metrics['climb_rate_fpm']:.0f} ft/min\n"
                f"  G-force: {metrics['g_force']:.1f}\n"
                f"  Ammo: {self.ammo_remaining}/200 "
                f"(37mm: {self.cannon_37mm_rounds}/40, 23mm: "
                f"{self.cannon_23mm_count * self.cannon_23mm_rounds_per_gun}/160)")
