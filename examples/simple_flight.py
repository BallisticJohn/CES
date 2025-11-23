#!/usr/bin/env python3
"""
Simple flight demonstration
Shows basic aircraft physics simulation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulator.aircraft.f86 import F86Sabre
from simulator.aircraft.mig15 import MiG15
from simulator.physics.vector import Vector3
from simulator.physics.quaternion import Quaternion
from simulator.physics.atmosphere import Atmosphere
import numpy as np


def main():
    print("=" * 60)
    print("Air Combat Simulator - Simple Flight Demo")
    print("=" * 60)
    print()

    # Create F-86 Sabre
    print("Creating F-86 Sabre...")
    altitude = Atmosphere.feet_to_meters(20000)  # 20,000 ft
    speed = Atmosphere.mph_to_mps(500)  # 500 mph

    f86 = F86Sabre(
        position=Vector3(0, 0, -altitude),
        velocity=Vector3(speed, 0, 0),
        orientation=Quaternion.identity()
    )

    print(f"Initial state:")
    print(f"  Altitude: {Atmosphere.meters_to_feet(f86.get_altitude()):.0f} ft")
    print(f"  Airspeed: {Atmosphere.mps_to_mph(f86.get_airspeed()):.0f} mph")
    print(f"  Mach: {f86.get_mach_number():.3f}")
    print()

    # Simulate for 10 seconds
    print("Simulating 10 seconds of flight...")
    dt = 0.05  # 50ms timesteps
    duration = 10.0

    time_history = []
    altitude_history = []
    speed_history = []

    time = 0.0
    while time < duration:
        time_history.append(time)
        altitude_history.append(f86.get_altitude())
        speed_history.append(f86.get_airspeed())

        f86.update(dt)
        time += dt

    print(f"Final state:")
    print(f"  Altitude: {Atmosphere.meters_to_feet(f86.get_altitude()):.0f} ft")
    print(f"  Airspeed: {Atmosphere.mps_to_mph(f86.get_airspeed()):.0f} mph")
    print(f"  Mach: {f86.get_mach_number():.3f}")
    print()

    # Performance metrics
    print("Performance metrics:")
    metrics = f86.get_performance_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
    print()

    # Create MiG-15 for comparison
    print("-" * 60)
    print("Creating MiG-15 for comparison...")
    mig = MiG15(
        position=Vector3(0, 0, -altitude),
        velocity=Vector3(speed, 0, 0),
        orientation=Quaternion.identity()
    )

    print(f"\nComparative specifications:")
    print(f"                      F-86 Sabre    MiG-15bis")
    print(f"  Mass:               {f86.mass:>6.0f} kg    {mig.mass:>6.0f} kg")
    print(f"  Wing area:          {f86.wing_area:>6.1f} m²    {mig.wing_area:>6.1f} m²")
    print(f"  Max thrust (SL):    {f86.max_thrust_sealevel/1000:>6.1f} kN    "
          f"{mig.max_thrust_sealevel/1000:>6.1f} kN")
    print(f"  Thrust/Weight:      {f86.max_thrust_sealevel/(f86.mass*9.81):>6.3f}       "
          f"{mig.max_thrust_sealevel/(mig.mass*9.81):>6.3f}")
    print(f"  Wing loading:       {f86.mass*9.81/f86.wing_area:>6.0f} Pa    "
          f"{mig.mass*9.81/mig.wing_area:>6.0f} Pa")
    print()

    # Calculate some performance parameters
    test_altitude = Atmosphere.feet_to_meters(30000)
    test_speed = Atmosphere.mph_to_mps(550)

    print(f"Performance at 30,000 ft, 550 mph:")

    f86_thrust = f86.get_thrust(1.0, test_altitude,
                                Atmosphere.get_mach_number(test_speed, test_altitude))
    mig_thrust = mig.get_thrust(1.0, test_altitude,
                                Atmosphere.get_mach_number(test_speed, test_altitude))

    print(f"  F-86 thrust:   {f86_thrust/1000:>6.1f} kN")
    print(f"  MiG-15 thrust: {mig_thrust/1000:>6.1f} kN")
    print()

    print("Specific excess power (Ps) at various altitudes:")
    for alt_ft in [10000, 20000, 30000, 40000]:
        alt_m = Atmosphere.feet_to_meters(alt_ft)
        f86_ps = f86.get_specific_excess_power(alt_m, test_speed)
        mig_ps = mig.get_specific_excess_power(alt_m, test_speed)
        print(f"  {alt_ft:>5} ft - F-86: {f86_ps:>6.1f} m/s, MiG-15: {mig_ps:>6.1f} m/s")

    print()
    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
