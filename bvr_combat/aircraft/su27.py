"""
Su-27 Flanker
"""
import numpy as np
from .base import Aircraft
from ..utils import Vector3


class Su27(Aircraft):
    """
    Su-27 Flanker (Su-27S/P/SM variants)

    Characteristics based on publicly available data
    """

    def __init__(self, position: Vector3, velocity: Vector3, name: str = "Su-27"):
        super().__init__(name, position, velocity)

        # Performance
        self.max_speed = 730.0  # m/s (~Mach 2.35 at altitude)
        self.cruise_speed = 250.0  # m/s (~Mach 0.73)
        self.max_turn_rate = np.radians(20)  # ~20 deg/s sustained (very agile)
        self.max_acceleration = 60.0  # m/s² (excellent thrust-to-weight)

        # RCS (Radar Cross Section)
        # Su-27 is a large aircraft with less RCS reduction than Western designs
        self.rcs_frontal = 15.0  # m² (large nose, intakes)
        self.rcs_side = 25.0  # m² (large wingspan, twin engines)
        self.rcs_rear = 20.0  # m² (twin engines, large tail)

        # N001 Zhuk Radar (older variants) or N001VE (upgraded)
        # Detection range: ~100-120 km vs 5m² target (look-up mode)
        # Better than MiG-29 but still inferior to F-16 APG-68
        self.radar_power = 1100.0  # Relative power (better than MiG-29)
        self.base_detection_range = 110000  # 110 km vs 5m² target

        # Weapons loadout (typical air superiority)
        # Su-27 can carry significantly more missiles than MiG-29
        self.max_missiles = 10  # Mix of R-73 (IR) + R-77 (ARH) + R-27 (SARH)
        self.missile_type = "R-77"

        # Avionics
        self.has_datalink = True  # Modern variants have limited datalink
        self.has_rwr = True  # Radar Warning Receiver
        self.has_ecm = True  # ECM (less capable than Western)

    def get_detection_range(self, target_rcs: float) -> float:
        """
        Su-27 N001/N001VE detection range

        Scales from known 110 km @ 5m² target
        """
        # Radar equation: R ∝ RCS^0.25
        reference_rcs = 5.0
        reference_range = self.base_detection_range

        return reference_range * (target_rcs / reference_rcs) ** 0.25
