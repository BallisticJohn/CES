"""
MiG-29 Fulcrum
"""
import numpy as np
from .base import Aircraft
from ..utils import Vector3


class MiG29(Aircraft):
    """
    MiG-29S Fulcrum

    Characteristics based on publicly available data
    """

    def __init__(self, position: Vector3, velocity: Vector3, name: str = "MiG-29"):
        super().__init__(name, position, velocity)

        # Performance
        self.max_speed = 680.0  # m/s (~Mach 2.0 at altitude)
        self.cruise_speed = 250.0  # m/s (~Mach 0.73)
        self.max_turn_rate = np.radians(20)  # ~20 deg/s sustained (very agile)
        self.max_acceleration = 55.0  # m/s² (good but slightly less than F-16)

        # RCS (Radar Cross Section)
        # MiG-29 has larger RCS due to less refined design
        self.rcs_frontal = 15.0  # m² (frontal - large intakes, less shaping)
        self.rcs_side = 20.0  # m²
        self.rcs_rear = 12.0  # m² (twin engines, nozzles)

        # N019 Zhuk Radar (older variants) or N010 Zhuk-M (upgraded)
        # Detection range: ~70-80 km vs 5m² target (look-up mode)
        # Inferior to Western radars of same generation
        self.radar_power = 900.0  # Relative power (less than F-16)
        self.base_detection_range = 75000  # 75 km vs 5m² target

        # Weapons loadout (typical air superiority)
        self.max_missiles = 6  # R-73 (IR) + R-77 (ARH)
        self.missile_type = "R-77"

        # Avionics
        self.has_datalink = False  # Limited/no Link-16 equivalent
        self.has_rwr = True  # Radar Warning Receiver
        self.has_ecm = True  # Basic ECM (less capable than Western)

    def get_detection_range(self, target_rcs: float) -> float:
        """
        MiG-29 N019/N010 detection range

        Scales from known 75 km @ 5m² target
        """
        # Radar equation: R ∝ RCS^0.25
        reference_rcs = 5.0
        reference_range = self.base_detection_range

        return reference_range * (target_rcs / reference_rcs) ** 0.25
