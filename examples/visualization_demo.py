#!/usr/bin/env python3
"""
Visualization demonstration
Shows energy-maneuverability diagrams and performance comparisons
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulator.aircraft.f86 import F86Sabre
from simulator.aircraft.mig15 import MiG15
from simulator.visualization import PerformancePlotter
from simulator.physics.atmosphere import Atmosphere


def main():
    print("=" * 60)
    print("Energy-Maneuverability Visualization Demo")
    print("=" * 60)
    print()

    # Create aircraft
    print("Creating aircraft...")
    f86 = F86Sabre()
    mig = MiG15()

    print("\n1. F-86 Sabre Energy-Maneuverability Diagram")
    print("-" * 60)
    print("Generating E-M diagram for F-86...")
    print("(Close the plot window to continue)")
    PerformancePlotter.plot_energy_maneuverability(f86)

    print("\n2. MiG-15 Energy-Maneuverability Diagram")
    print("-" * 60)
    print("Generating E-M diagram for MiG-15...")
    print("(Close the plot window to continue)")
    PerformancePlotter.plot_energy_maneuverability(mig)

    print("\n3. F-86 Turn Performance")
    print("-" * 60)
    print("Generating turn performance charts for F-86...")
    print("(Close the plot window to continue)")
    PerformancePlotter.plot_turn_performance(f86)

    print("\n4. MiG-15 Turn Performance")
    print("-" * 60)
    print("Generating turn performance charts for MiG-15...")
    print("(Close the plot window to continue)")
    PerformancePlotter.plot_turn_performance(mig)

    print("\n5. Comparative Performance Analysis")
    print("-" * 60)
    print("Comparing F-86 vs MiG-15 at 25,000 ft...")
    print("(Close the plot window to continue)")
    PerformancePlotter.plot_comparative_performance(f86, mig)

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print()
    print("Key observations:")
    print("- MiG-15 has superior Ps at all altitudes (better T/W ratio)")
    print("- F-86 has better transonic performance")
    print("- MiG-15 has better sustained turn rate, especially at altitude")
    print("- F-86 has advantage in high-speed, low-altitude regimes")


if __name__ == '__main__':
    main()
