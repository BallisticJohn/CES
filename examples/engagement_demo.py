#!/usr/bin/env python3
"""
Engagement demonstration
Shows F-86 vs MiG-15 dogfight simulation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulator.aircraft.f86 import F86Sabre
from simulator.aircraft.mig15 import MiG15
from simulator.physics.vector import Vector3
from simulator.physics.quaternion import Quaternion
from simulator.physics.atmosphere import Atmosphere
from simulator.simulation.engagement import EngagementSimulation
import numpy as np


def main():
    print("=" * 60)
    print("Air Combat Simulator - Engagement Demo")
    print("F-86 Sabre vs MiG-15bis")
    print("=" * 60)
    print()

    # Initial conditions: Co-altitude, converging
    altitude = Atmosphere.feet_to_meters(25000)  # 25,000 ft
    speed = Atmosphere.mph_to_mps(500)  # 500 mph both

    # F-86 starts at origin, heading east
    f86 = F86Sabre(
        position=Vector3(0, 0, -altitude),
        velocity=Vector3(speed, 0, 0),
        orientation=Quaternion.from_euler(0, 0, 0)
    )

    # MiG-15 starts 10km east, heading west (toward F-86)
    mig = MiG15(
        position=Vector3(10000, 0, -altitude),
        velocity=Vector3(-speed, 0, 0),
        orientation=Quaternion.from_euler(0, 0, np.pi)  # 180 degrees
    )

    print("Initial Setup:")
    print(f"  Altitude: {Atmosphere.meters_to_feet(altitude):.0f} ft")
    print(f"  Speed: {Atmosphere.mps_to_mph(speed):.0f} mph")
    print(f"  Initial separation: 10,000 m")
    print(f"  Geometry: Head-on pass")
    print()

    # Create and run engagement
    print("Running engagement simulation...")
    sim = EngagementSimulation(f86, mig)
    result = sim.run(max_time=60.0, dt=0.05)  # 1 minute max

    print()
    print("Engagement Results:")
    print(f"  Duration: {result['duration']:.1f} seconds")
    print(f"  Outcome: {result.get('winner', 'Draw')}")
    print(f"  Reason: {result.get('reason', 'N/A')}")
    print(f"  Final separation: {result.get('final_separation', 0):.0f} m")
    print()

    # Show some statistics from the engagement
    print("Engagement Statistics:")
    print(f"  Number of timesteps: {len(sim.history['time'])}")

    # Minimum separation
    separations = [sim.get_separation(i) for i in range(len(sim.history['time']))]
    min_sep = min(separations)
    min_sep_time = sim.history['time'][separations.index(min_sep)]
    print(f"  Closest approach: {min_sep:.0f} m at t={min_sep_time:.1f} s")

    # Get aspect angles at closest approach
    min_idx = separations.index(min_sep)
    aspect1, aspect2 = sim.get_aspect_angle(min_idx)
    print(f"  Aspect angles at closest:")
    print(f"    F-86 to MiG-15: {aspect1:.1f}°")
    print(f"    MiG-15 to F-86: {aspect2:.1f}°")

    print()

    # Final states
    print("Final Aircraft States:")
    print()
    print("F-86 Sabre:")
    f86_final = sim.aircraft1.get_performance_metrics()
    print(f"  Altitude: {f86_final['altitude_ft']:.0f} ft")
    print(f"  Airspeed: {f86_final['airspeed_mph']:.0f} mph (Mach {f86_final['mach']:.2f})")
    print(f"  Position: {sim.aircraft1.state.position}")

    print()
    print("MiG-15bis:")
    mig_final = sim.aircraft2.get_performance_metrics()
    print(f"  Altitude: {mig_final['altitude_ft']:.0f} ft")
    print(f"  Airspeed: {mig_final['airspeed_mph']:.0f} mph (Mach {mig_final['mach']:.2f})")
    print(f"  Position: {sim.aircraft2.state.position}")

    print()
    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print()
    print("Note: This is a basic physics simulation without AI control,")
    print("weapon systems, or damage modeling. Aircraft fly ballistically")
    print("after initial conditions. Full AI and combat features coming soon!")


if __name__ == '__main__':
    main()
