#!/usr/bin/env python3
"""
Enhanced engagement visualization
Shows trajectories and detailed analysis
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
from simulator.visualization import TrajectoryPlotter, EngagementAnalyzer
import numpy as np


def main():
    print("=" * 60)
    print("Enhanced Engagement Visualization Demo")
    print("=" * 60)
    print()

    # Setup: Head-on pass at 25,000 ft
    altitude = Atmosphere.feet_to_meters(25000)
    speed = Atmosphere.mph_to_mps(500)

    print("Scenario: Head-on pass")
    print(f"  Altitude: 25,000 ft")
    print(f"  Speed: 500 mph both aircraft")
    print(f"  Initial separation: 10 km")
    print()

    # F-86 heading east
    f86 = F86Sabre(
        position=Vector3(0, 0, -altitude),
        velocity=Vector3(speed, 0, 0),
        orientation=Quaternion.from_euler(0, 0, 0)
    )

    # MiG-15 heading west
    mig = MiG15(
        position=Vector3(10000, 0, -altitude),
        velocity=Vector3(-speed, 0, 0),
        orientation=Quaternion.from_euler(0, 0, np.pi)
    )

    # Run engagement
    print("Running engagement simulation...")
    sim = EngagementSimulation(f86, mig)
    result = sim.run(max_time=60.0, dt=0.05)

    print(f"Duration: {result['duration']:.1f} seconds")
    print(f"Closest approach: {result.get('final_separation', 0):.0f} m")
    print()

    # 3D Trajectory (matplotlib)
    print("1. 3D Trajectory Plot (matplotlib)")
    print("-" * 60)
    print("Generating 3D trajectory with velocity vectors...")
    print("(Close the plot window to continue)")
    TrajectoryPlotter.plot_3d_trajectory(sim, interactive=False,
                                        show_velocity_vectors=True)

    # Engagement analysis
    print("\n2. Detailed Engagement Analysis")
    print("-" * 60)
    print("Generating comprehensive analysis plots...")
    print("(Close the plot window to continue)")
    EngagementAnalyzer.plot_engagement_analysis(sim)

    # Interactive 3D (plotly)
    print("\n3. Interactive 3D Trajectory (plotly)")
    print("-" * 60)
    print("Generating interactive plot (will open in browser)...")
    print("(Close the browser tab to continue)")
    TrajectoryPlotter.plot_3d_trajectory(sim, interactive=True)

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
