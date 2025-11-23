"""
Aerodynamic coefficient models for aircraft
Based on wind tunnel data and empirical models
"""

import numpy as np
from typing import Dict, Callable


class AeroCoefficients:
    """
    Aerodynamic coefficients for aircraft
    All coefficients are dimensionless
    """

    def __init__(self, config: Dict):
        """
        Initialize aerodynamic coefficients

        Args:
            config: Dictionary containing coefficient data or functions
        """
        self.config = config

    @staticmethod
    def simple_lift_coefficient(alpha: float, CL0: float = 0.0, CLalpha: float = 5.5,
                                 alpha_stall: float = 0.26) -> float:
        """
        Simple lift coefficient model

        Args:
            alpha: Angle of attack (radians)
            CL0: Zero-alpha lift coefficient
            CLalpha: Lift curve slope (per radian)
            alpha_stall: Stall angle (radians)

        Returns:
            Lift coefficient CL
        """
        # Linear region
        if abs(alpha) < alpha_stall:
            return CL0 + CLalpha * alpha

        # Post-stall region (simplified)
        sign = np.sign(alpha)
        CL_stall = CL0 + CLalpha * alpha_stall * sign

        # Gradual reduction after stall
        alpha_excess = abs(alpha) - alpha_stall
        stall_factor = np.exp(-2.0 * alpha_excess)

        return CL_stall * stall_factor

    @staticmethod
    def simple_drag_coefficient(CL: float, CD0: float = 0.02, K: float = 0.05) -> float:
        """
        Simple drag coefficient model (parabolic drag polar)

        Args:
            CL: Lift coefficient
            CD0: Zero-lift drag coefficient (parasitic drag)
            K: Induced drag factor (related to aspect ratio)

        Returns:
            Drag coefficient CD
        """
        # CD = CD0 + K * CL²
        return CD0 + K * CL * CL

    @staticmethod
    def transonic_drag_rise(mach: float, CD_subsonic: float,
                            mach_crit: float = 0.8, mach_drag_div: float = 0.9) -> float:
        """
        Transonic drag rise model (wave drag)

        Args:
            mach: Mach number
            CD_subsonic: Subsonic drag coefficient
            mach_crit: Critical Mach number (drag rise begins)
            mach_drag_div: Drag divergence Mach number (severe rise)

        Returns:
            Total drag coefficient including wave drag
        """
        if mach < mach_crit:
            return CD_subsonic

        if mach < mach_drag_div:
            # Gradual rise
            delta_mach = mach - mach_crit
            rise_factor = (delta_mach / (mach_drag_div - mach_crit)) ** 2
            wave_drag = 0.1 * rise_factor
            return CD_subsonic + wave_drag
        else:
            # Severe drag rise in transonic region
            delta_mach = mach - mach_drag_div
            wave_drag = 0.1 + 0.3 * (1.0 - np.exp(-5.0 * delta_mach))
            return CD_subsonic + wave_drag

    @staticmethod
    def side_force_coefficient(beta: float, CYbeta: float = -0.7) -> float:
        """
        Side force coefficient

        Args:
            beta: Sideslip angle (radians)
            CYbeta: Side force derivative (per radian)

        Returns:
            Side force coefficient CY
        """
        return CYbeta * beta

    @staticmethod
    def rolling_moment_coefficient(p: float, beta: float,
                                   Clp: float = -0.4, Clbeta: float = -0.1) -> float:
        """
        Rolling moment coefficient

        Args:
            p: Roll rate (rad/s)
            beta: Sideslip angle (radians)
            Clp: Roll damping derivative
            Clbeta: Dihedral effect

        Returns:
            Rolling moment coefficient Cl
        """
        return Clp * p + Clbeta * beta

    @staticmethod
    def pitching_moment_coefficient(alpha: float, q: float, elevator: float = 0.0,
                                    CM0: float = 0.0, CMalpha: float = -0.5,
                                    CMq: float = -10.0, CMde: float = -1.0) -> float:
        """
        Pitching moment coefficient

        Args:
            alpha: Angle of attack (radians)
            q: Pitch rate (rad/s)
            elevator: Elevator deflection (radians)
            CM0: Zero-alpha pitching moment
            CMalpha: Pitch stiffness derivative
            CMq: Pitch damping derivative
            CMde: Elevator effectiveness

        Returns:
            Pitching moment coefficient Cm
        """
        return CM0 + CMalpha * alpha + CMq * q + CMde * elevator

    @staticmethod
    def yawing_moment_coefficient(beta: float, r: float, rudder: float = 0.0,
                                  CNbeta: float = 0.12, CNr: float = -0.15,
                                  CNdr: float = -0.1) -> float:
        """
        Yawing moment coefficient

        Args:
            beta: Sideslip angle (radians)
            r: Yaw rate (rad/s)
            rudder: Rudder deflection (radians)
            CNbeta: Directional stability derivative
            CNr: Yaw damping derivative
            CNdr: Rudder effectiveness

        Returns:
            Yawing moment coefficient Cn
        """
        return CNbeta * beta + CNr * r + CNdr * rudder


class F86AeroCoefficients(AeroCoefficients):
    """
    F-86 Sabre specific aerodynamic coefficients
    Based on wind tunnel data and flight test results
    """

    def __init__(self):
        config = {
            'CL0': 0.15,           # Zero-alpha lift coefficient
            'CLalpha': 5.7,        # Lift curve slope (per radian)
            'alpha_stall': 0.28,   # ~16 degrees
            'CD0': 0.018,          # Zero-lift drag
            'K': 0.048,            # Induced drag factor (e ~ 0.8)
            'mach_crit': 0.85,     # Critical Mach number
            'mach_drag_div': 0.91, # Drag divergence Mach
            'CM0': -0.02,          # Pitch moment at zero alpha
            'CMalpha': -0.6,       # Pitch stiffness
            'aspect_ratio': 4.79,  # Wing aspect ratio
        }
        super().__init__(config)

    def get_CL(self, alpha: float, mach: float) -> float:
        """Get lift coefficient for F-86"""
        CL_base = self.simple_lift_coefficient(
            alpha,
            self.config['CL0'],
            self.config['CLalpha'],
            self.config['alpha_stall']
        )

        # Compressibility correction (Prandtl-Glauert)
        if mach < 0.7:
            beta = np.sqrt(1.0 - mach ** 2)
            return CL_base / beta
        else:
            # Empirical correction for transonic
            return CL_base * (1.0 + 0.1 * (0.9 - mach))

    def get_CD(self, CL: float, mach: float) -> float:
        """Get drag coefficient for F-86"""
        CD_sub = self.simple_drag_coefficient(
            CL,
            self.config['CD0'],
            self.config['K']
        )

        return self.transonic_drag_rise(
            mach,
            CD_sub,
            self.config['mach_crit'],
            self.config['mach_drag_div']
        )


class MiG15AeroCoefficients(AeroCoefficients):
    """
    MiG-15 specific aerodynamic coefficients
    Based on Russian wind tunnel data and defector aircraft testing
    """

    def __init__(self):
        config = {
            'CL0': 0.18,           # Slightly higher due to wing design
            'CLalpha': 5.5,        # Similar to F-86
            'alpha_stall': 0.24,   # ~14 degrees (earlier stall)
            'CD0': 0.019,          # Slightly higher parasitic drag
            'K': 0.055,            # Higher induced drag (lower AR)
            'mach_crit': 0.82,     # Lower critical Mach
            'mach_drag_div': 0.88, # Earlier drag rise
            'CM0': -0.03,          # More nose-heavy
            'CMalpha': -0.55,      # Slightly less stable
            'aspect_ratio': 4.93,  # Wing aspect ratio
        }
        super().__init__(config)

    def get_CL(self, alpha: float, mach: float) -> float:
        """Get lift coefficient for MiG-15"""
        CL_base = self.simple_lift_coefficient(
            alpha,
            self.config['CL0'],
            self.config['CLalpha'],
            self.config['alpha_stall']
        )

        # Compressibility correction
        if mach < 0.7:
            beta = np.sqrt(1.0 - mach ** 2)
            return CL_base / beta
        else:
            # MiG-15 had poorer transonic characteristics
            return CL_base * (1.0 + 0.15 * (0.85 - mach))

    def get_CD(self, CL: float, mach: float) -> float:
        """Get drag coefficient for MiG-15"""
        CD_sub = self.simple_drag_coefficient(
            CL,
            self.config['CD0'],
            self.config['K']
        )

        return self.transonic_drag_rise(
            mach,
            CD_sub,
            self.config['mach_crit'],
            self.config['mach_drag_div']
        )
