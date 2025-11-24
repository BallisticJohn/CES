"""
F-16C Fighting Falcon
"""
import numpy as np
from .base import Aircraft
from ..utils import Vector3


class F16C(Aircraft):
    """
    F-16C Block 50/52 Fighting Falcon

    Characteristics based on publicly available data
    """

    def __init__(self, position: Vector3, velocity: Vector3, name: str = "F-16C"):
        super().__init__(name, position, velocity)

        # Performance
        self.max_speed = 680.0  # m/s (~Mach 2.0 at altitude)
        self.cruise_speed = 240.0  # m/s (~Mach 0.7)
        self.max_turn_rate = np.radians(18)  # ~18 deg/s sustained
        self.max_acceleration = 60.0  # m/s² (good thrust-to-weight)

        # RCS (Radar Cross Section)
        # F-16 has relatively small RCS for 4th gen
        self.rcs_frontal = 5.0  # m² (clean configuration, frontal)
        self.rcs_side = 8.0  # m²
        self.rcs_rear = 3.0  # m² (small tail cross-section)

        # AN/APG-68(V)9 Radar
        # Detection range: ~105 km vs 5m² target (look-up mode)
        self.radar_power = 1200.0  # Relative power
        self.base_detection_range = 105000  # 105 km vs 5m² target

        # Weapons loadout (typical air superiority)
        self.max_missiles = 6  # 2x wingtip, 4x underwing
        self.missile_type = "AIM-120C"

        # Avionics
        self.has_datalink = True  # Link-16
        self.has_rwr = True  # Radar Warning Receiver
        self.has_ecm = True  # Electronic Countermeasures

    def get_detection_range(self, target_rcs: float) -> float:
        """
        F-16 APG-68 detection range

        Scales from known 105 km @ 5m² target
        """
        # Radar equation: R ∝ RCS^0.25
        reference_rcs = 5.0
        reference_range = self.base_detection_range

        return reference_range * (target_rcs / reference_rcs) ** 0.25
