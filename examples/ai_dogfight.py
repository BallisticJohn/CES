#!/usr/bin/env python3
"""
AI vs AI dogfight demonstration
Watch two AI pilots engage in realistic BFM
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
from simulator.ai import PilotAI
from simulator.visualization import TrajectoryPlotter, EngagementAnalyzer
import numpy as np


class AIEngagementSimulation(EngagementSimulation):
    """Extended engagement simulation with AI pilots"""

    def __init__(self, aircraft1, aircraft2, pilot1: PilotAI, pilot2: PilotAI):
        super().__init__(aircraft1, aircraft2)
        self.pilot1 = pilot1
        self.pilot2 = pilot2

        # Set targets
        self.pilot1.set_target(aircraft2)
        self.pilot2.set_target(aircraft1)

    def run(self, max_time: float = None, dt: float = None) -> dict:
        """Run AI-controlled engagement"""
        if max_time is not None:
            self.max_time = max_time
        if dt is not None:
            self.dt = dt

        time = 0.0
        self.history['time'] = []

        print(f"{'Time':>6} {'F-86 Maneuver':<20} {'MiG-15 Maneuver':<20} {'Range':>8} {'Separation':>10}")
        print("-" * 80)

        while time < self.max_time:
            # Record state
            self._record_state(time)

            # Update AI pilots
            self.pilot1.update(self.dt)
            self.pilot2.update(self.dt)

            # Update aircraft physics
            self.aircraft1.update(self.dt)
            self.aircraft2.update(self.dt)

            # Print status every 5 seconds
            if int(time) % 5 == 0 and abs(time - int(time)) < self.dt:
                tactical1 = self.pilot1._assess_situation()
                range_nm = tactical1['range'] / 1852  # nautical miles
                separation_m = self.get_separation(-1)

                print(f"{time:6.1f}s {self.pilot1.current_maneuver:<20} "
                      f"{self.pilot2.current_maneuver:<20} "
                      f"{range_nm:7.2f}nm {separation_m:8.0f}m")

            # Check termination
            if self._check_termination():
                break

            time += self.dt

        print("-" * 80)

        # Compile results
        self.result = self._compile_results()
        return self.result


def main():
    print("=" * 80)
    print("AI vs AI DOGFIGHT - F-86 Sabre vs MiG-15")
    print("=" * 80)
    print()

    # Scenario setup
    scenarios = {
        '1': {
            'name': 'Head-on Pass',
            'description': 'Classic head-on merge, both aircraft at equal energy',
            'f86_pos': Vector3(0, 0, -Atmosphere.feet_to_meters(25000)),
            'f86_vel': Vector3(Atmosphere.mph_to_mps(500), 0, 0),
            'f86_hdg': 0,
            'mig_pos': Vector3(8000, 0, -Atmosphere.feet_to_meters(25000)),
            'mig_vel': Vector3(-Atmosphere.mph_to_mps(500), 0, 0),
            'mig_hdg': np.pi,
        },
        '2': {
            'name': 'Perch Setup',
            'description': 'F-86 starts behind and high (classic bounce)',
            'f86_pos': Vector3(-3000, 1000, -Atmosphere.feet_to_meters(28000)),
            'f86_vel': Vector3(Atmosphere.mph_to_mps(480), 0, 0),
            'f86_hdg': 0,
            'mig_pos': Vector3(0, 0, -Atmosphere.feet_to_meters(25000)),
            'mig_vel': Vector3(Atmosphere.mph_to_mps(450), 0, 0),
            'mig_hdg': 0,
        },
        '3': {
            'name': 'Crossing Shot',
            'description': '90-degree crossing engagement',
            'f86_pos': Vector3(0, -4000, -Atmosphere.feet_to_meters(26000)),
            'f86_vel': Vector3(Atmosphere.mph_to_mps(480), 0, 0),
            'f86_hdg': 0,
            'mig_pos': Vector3(0, 0, -Atmosphere.feet_to_meters(26000)),
            'mig_vel': Vector3(0, Atmosphere.mph_to_mps(480), 0),
            'mig_hdg': np.pi/2,
        },
    }

    # Select scenario
    print("Select scenario:")
    for key, scenario in scenarios.items():
        print(f"  {key}. {scenario['name']} - {scenario['description']}")
    print()

    choice = input("Enter scenario number (1-3, or Enter for 1): ").strip()
    if choice not in scenarios:
        choice = '1'

    scenario = scenarios[choice]
    print(f"\nRunning: {scenario['name']}")
    print(f"{scenario['description']}")
    print()

    # Create aircraft
    f86 = F86Sabre(
        position=scenario['f86_pos'],
        velocity=scenario['f86_vel'],
        orientation=Quaternion.from_euler(0, 0, scenario['f86_hdg'])
    )

    mig = MiG15(
        position=scenario['mig_pos'],
        velocity=scenario['mig_vel'],
        orientation=Quaternion.from_euler(0, 0, scenario['mig_hdg'])
    )

    # Create AI pilots
    # F-86 pilot: skilled, moderate aggression
    pilot_f86 = PilotAI(f86, aggression=0.7, skill_level=0.85)

    # MiG-15 pilot: skilled, aggressive (uses energy advantage)
    pilot_mig = PilotAI(mig, aggression=0.8, skill_level=0.85)

    print("F-86 Pilot: Skill=0.85, Aggression=0.7")
    print("MiG-15 Pilot: Skill=0.85, Aggression=0.8")
    print()
    print("Starting engagement...")
    print()

    # Run engagement (smaller timestep for stability)
    sim = AIEngagementSimulation(f86, mig, pilot_f86, pilot_mig)
    result = sim.run(max_time=120.0, dt=0.02)

    print()
    print("=" * 80)
    print("ENGAGEMENT COMPLETE")
    print("=" * 80)
    print(f"Duration: {result['duration']:.1f} seconds")
    print(f"Outcome: {result.get('winner', 'Draw')}")
    print(f"Reason: {result.get('reason', 'N/A')}")
    print()

    # Final pilot status
    print("Final Status:")
    print("-" * 80)
    tactical_f86 = pilot_f86._assess_situation()
    tactical_mig = pilot_mig._assess_situation()

    print("F-86 Sabre:")
    print(pilot_f86.get_status_string(tactical_f86))
    print()
    print("MiG-15:")
    print(pilot_mig.get_status_string(tactical_mig))
    print()

    # Visualize
    print("Generating visualizations...")
    print("(Close each plot window to continue)")

    # 3D trajectory
    TrajectoryPlotter.plot_3d_trajectory(sim, interactive=False,
                                        show_velocity_vectors=True)

    # Detailed analysis
    EngagementAnalyzer.plot_engagement_analysis(sim)

    # Interactive 3D
    response = input("\nGenerate interactive 3D plot? (y/n): ").strip().lower()
    if response == 'y':
        TrajectoryPlotter.plot_3d_trajectory(sim, interactive=True)

    print()
    print("=" * 80)
    print("Demo complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
