#!/usr/bin/env python3
"""
1v1 BVR engagement test: F-16C vs MiG-29

Demonstrates the simplified BVR combat model
"""
import sys
import os

# Add parent directory to path so imports work from any location
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from bvr_combat.aircraft.f16 import F16C
from bvr_combat.aircraft.mig29 import MiG29
from bvr_combat.engagement.battle import Engagement
from bvr_combat.utils import Vector3, feet_to_meters, knots_to_mps
from bvr_combat.visualization import plot_engagement, plot_timeline
import matplotlib.pyplot as plt


def main():
    print("\n" + "=" * 80)
    print("F-16C vs MiG-29 - 1v1 BVR ENGAGEMENT")
    print("=" * 80)
    print()

    # Scenario: Head-on engagement at 40,000 ft
    altitude = -feet_to_meters(40000)  # Negative Z = altitude
    speed = knots_to_mps(450)  # 450 knots cruise

    # F-16C (Blue) - starting from west
    f16_pos = Vector3(0, 0, altitude)
    f16_vel = Vector3(speed, 0, 0)  # Flying east
    f16 = F16C(f16_pos, f16_vel, name="Viper-1")

    # Load missiles
    f16.missiles = ["AIM-120C"] * 4  # 4x AMRAAM

    # MiG-29 (Red) - starting from east, flying west
    separation = 120000  # 120 km initial separation
    mig_pos = Vector3(separation, 0, altitude)
    mig_vel = Vector3(-speed, 0, 0)  # Flying west (toward F-16)
    mig = MiG29(mig_pos, mig_vel, name="Fulcrum-1")

    # Load missiles
    mig.missiles = ["R-77"] * 4  # 4x R-77

    # Print initial setup
    print("Initial Setup:")
    print("-" * 80)
    print(f"  F-16C: {f16}")
    print(f"    Position: {f16.position}")
    print(f"    Speed: {f16.get_speed():.0f} m/s ({f16.get_speed()/0.514444:.0f} kts)")
    print(f"    Missiles: {len(f16.missiles)}x {f16.missile_type}")
    print(f"    Radar: {f16.base_detection_range/1000:.0f} km vs 5m² target")
    print()
    print(f"  MiG-29: {mig}")
    print(f"    Position: {mig.position}")
    print(f"    Speed: {mig.get_speed():.0f} m/s ({mig.get_speed()/0.514444:.0f} kts)")
    print(f"    Missiles: {len(mig.missiles)}x {mig.missile_type}")
    print(f"    Radar: {mig.base_detection_range/1000:.0f} km vs 5m² target")
    print()

    # Calculate detection ranges
    print("Detection Ranges:")
    print("-" * 80)
    f16_rcs = f16.get_rcs(mig.position)
    mig_rcs = mig.get_rcs(f16.position)

    f16_detect_range = f16.get_detection_range(mig_rcs)
    mig_detect_range = mig.get_detection_range(f16_rcs)

    print(f"  F-16 RCS (from MiG): {f16_rcs:.1f} m²")
    print(f"  MiG-29 RCS (from F-16): {mig_rcs:.1f} m²")
    print()
    print(f"  F-16 can detect MiG at: {f16_detect_range/1000:.1f} km")
    print(f"  MiG can detect F-16 at: {mig_detect_range/1000:.1f} km")
    print()
    print(f"  Detection advantage: F-16 by {(f16_detect_range - mig_detect_range)/1000:.1f} km")
    print()

    # Create engagement
    blue_force = [f16]
    red_force = [mig]

    engagement = Engagement(blue_force, red_force, dt=1.0)

    # Debug: check key moments
    print("Debug: Watching engagement develop")
    print("-" * 80)
    last_print = 0
    for i in range(600):
        engagement.step()
        range_between = f16.position.distance_to(mig.position)

        # Print every 10 seconds or when something interesting happens
        if engagement.time - last_print >= 10 or len(engagement.missiles) > 0:
            f16_can_see = f16.can_detect(mig)
            mig_can_see = mig.can_detect(f16)
            print(f"t={engagement.time:5.0f}s  Range:{range_between/1000:6.1f}km  "
                  f"F16:{f16_can_see}  MiG:{mig_can_see}  Missiles:{len(engagement.missiles)}")
            last_print = engagement.time

        if not engagement.engagement_active:
            print(f"Engagement ended at t={engagement.time:.0f}s")
            break

    print()

    # Get results
    results = {
        'duration': engagement.time,
        'blue_survivors': sum(1 for a in blue_force if a.alive),
        'red_survivors': sum(1 for a in red_force if a.alive),
        'blue_kills': engagement.blue_kills,
        'red_kills': engagement.red_kills
    }

    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Duration: {results['duration']:.1f} seconds")
    print(f"Blue survivors: {results['blue_survivors']}/{len(blue_force)}")
    print(f"Red survivors: {results['red_survivors']}/{len(red_force)}")
    print(f"Blue kills: {results['blue_kills']}")
    print(f"Red kills: {results['red_kills']}")
    print()

    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Analyze why the winner won
    if results['blue_survivors'] > 0 and results['red_survivors'] == 0:
        print("F-16 victory due to:")
        print("  - Superior radar detection range (~35% advantage)")
        print("  - Longer-range missiles (AIM-120C: 105km vs R-77: 80km)")
        print("  - Shot first, killed MiG before it could effectively engage")

    elif results['red_survivors'] > 0 and results['blue_survivors'] == 0:
        print("MiG-29 victory!")
        print("  - Possible factors: luck, first-shot advantage, better geometry")

    else:
        print("Mutual kill or survival")

    print()

    # Generate visualizations
    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Engagement plot
    fig1 = plot_engagement(blue_force, red_force, engagement.events,
                          engagement.blue_history, engagement.red_history,
                          engagement.missile_history)

    # Timeline plot
    fig2 = plot_timeline(engagement.events, results['duration'])

    print("Displaying plots... (close windows to exit)")
    plt.show()


if __name__ == "__main__":
    main()
