"""
International Standard Atmosphere (ISA) model
Provides atmospheric properties as a function of altitude
"""

import numpy as np
from typing import Tuple


class Atmosphere:
    """
    Standard atmosphere model (1976 U.S. Standard Atmosphere)

    Valid up to 86 km altitude
    All values in SI units
    """

    # Sea level standard conditions
    P0 = 101325.0  # Pa (pressure)
    T0 = 288.15    # K (temperature)
    RHO0 = 1.225   # kg/m³ (density)

    # Gas constants
    R = 287.05     # J/(kg·K) - specific gas constant for air
    GAMMA = 1.4    # Ratio of specific heats
    G = 9.80665    # m/s² - gravitational acceleration

    # Atmospheric layers (altitude in meters, lapse rate in K/m)
    LAYERS = [
        (0,     -0.0065),   # Troposphere
        (11000,  0.0),      # Tropopause
        (20000,  0.001),    # Stratosphere 1
        (32000,  0.0028),   # Stratosphere 2
        (47000,  0.0),      # Stratopause
        (51000, -0.0028),   # Mesosphere 1
        (71000, -0.002),    # Mesosphere 2
    ]

    @staticmethod
    def get_properties(altitude: float) -> Tuple[float, float, float, float]:
        """
        Get atmospheric properties at given altitude

        Args:
            altitude: Geometric altitude in meters (above sea level)

        Returns:
            Tuple of (temperature, pressure, density, speed_of_sound)
            - temperature: K
            - pressure: Pa
            - density: kg/m³
            - speed_of_sound: m/s
        """
        # Handle below sea level
        if altitude < 0:
            altitude = 0

        # Find the appropriate atmospheric layer
        h = altitude
        T = Atmosphere.T0
        P = Atmosphere.P0

        for i, (h_base, lapse_rate) in enumerate(Atmosphere.LAYERS):
            if i + 1 < len(Atmosphere.LAYERS):
                h_top = Atmosphere.LAYERS[i + 1][0]
            else:
                h_top = 86000

            if h < h_top:
                # Calculate temperature
                if i == 0:
                    T_base = Atmosphere.T0
                    P_base = Atmosphere.P0
                else:
                    # Calculate base conditions for this layer
                    T_base, P_base, _, _ = Atmosphere._calculate_base_conditions(h_base)

                dh = h - h_base

                if abs(lapse_rate) < 1e-10:
                    # Isothermal layer
                    T = T_base
                    P = P_base * np.exp(-Atmosphere.G * dh / (Atmosphere.R * T_base))
                else:
                    # Gradient layer
                    T = T_base + lapse_rate * dh
                    P = P_base * (T / T_base) ** (-Atmosphere.G / (lapse_rate * Atmosphere.R))

                break

        # Calculate density from ideal gas law
        rho = P / (Atmosphere.R * T)

        # Calculate speed of sound
        a = np.sqrt(Atmosphere.GAMMA * Atmosphere.R * T)

        return T, P, rho, a

    @staticmethod
    def _calculate_base_conditions(h_base: float) -> Tuple[float, float, float, float]:
        """Calculate conditions at the base of a layer"""
        T = Atmosphere.T0
        P = Atmosphere.P0

        for i, (h_layer, lapse_rate) in enumerate(Atmosphere.LAYERS):
            if i + 1 < len(Atmosphere.LAYERS):
                h_top = Atmosphere.LAYERS[i + 1][0]
            else:
                break

            if h_base <= h_layer:
                break

            dh = min(h_top, h_base) - h_layer

            if abs(lapse_rate) < 1e-10:
                # Isothermal layer
                T_next = T
                P = P * np.exp(-Atmosphere.G * dh / (Atmosphere.R * T))
            else:
                # Gradient layer
                T_next = T + lapse_rate * dh
                P = P * (T_next / T) ** (-Atmosphere.G / (lapse_rate * Atmosphere.R))
                T = T_next

            if h_base <= h_top:
                break

        rho = P / (Atmosphere.R * T)
        a = np.sqrt(Atmosphere.GAMMA * Atmosphere.R * T)
        return T, P, rho, a

    @staticmethod
    def get_temperature(altitude: float) -> float:
        """Get temperature at altitude (K)"""
        return Atmosphere.get_properties(altitude)[0]

    @staticmethod
    def get_pressure(altitude: float) -> float:
        """Get pressure at altitude (Pa)"""
        return Atmosphere.get_properties(altitude)[1]

    @staticmethod
    def get_density(altitude: float) -> float:
        """Get density at altitude (kg/m³)"""
        return Atmosphere.get_properties(altitude)[2]

    @staticmethod
    def get_speed_of_sound(altitude: float) -> float:
        """Get speed of sound at altitude (m/s)"""
        return Atmosphere.get_properties(altitude)[3]

    @staticmethod
    def get_mach_number(velocity: float, altitude: float) -> float:
        """
        Calculate Mach number

        Args:
            velocity: True airspeed (m/s)
            altitude: Altitude (m)

        Returns:
            Mach number
        """
        a = Atmosphere.get_speed_of_sound(altitude)
        return velocity / a

    @staticmethod
    def get_dynamic_pressure(velocity: float, altitude: float) -> float:
        """
        Calculate dynamic pressure (q)

        Args:
            velocity: True airspeed (m/s)
            altitude: Altitude (m)

        Returns:
            Dynamic pressure (Pa)
        """
        rho = Atmosphere.get_density(altitude)
        return 0.5 * rho * velocity ** 2

    @staticmethod
    def feet_to_meters(feet: float) -> float:
        """Convert feet to meters"""
        return feet * 0.3048

    @staticmethod
    def meters_to_feet(meters: float) -> float:
        """Convert meters to feet"""
        return meters / 0.3048

    @staticmethod
    def knots_to_mps(knots: float) -> float:
        """Convert knots to meters per second"""
        return knots * 0.514444

    @staticmethod
    def mps_to_knots(mps: float) -> float:
        """Convert meters per second to knots"""
        return mps / 0.514444

    @staticmethod
    def mph_to_mps(mph: float) -> float:
        """Convert miles per hour to meters per second"""
        return mph * 0.44704

    @staticmethod
    def mps_to_mph(mps: float) -> float:
        """Convert meters per second to miles per hour"""
        return mps / 0.44704
