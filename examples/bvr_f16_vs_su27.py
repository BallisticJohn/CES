#!/usr/bin/env python3
"""
1v1 BVR engagement test: F-16C vs Su-27

Tests F-16 against a more capable adversary
"""
import sys
import os

# Add parent directory to path so imports work from any location
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from bvr_combat.aircraft.f16 import F16C
from bvr_combat.aircraft.su27 import Su27
from bvr_combat.engagement.battle import Engagement
from bvr_combat.utils import Vector3, feet_to_meters, knots_to_mps
from bvr_combat.visualization import plot_engagement, plot_timeline
import matplotlib.pyplot as plt


def main():
    print("\n" + "=" * 80)
    print("F-16C vs Su-27 - 1v1 BVR ENGAGEMENT")
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
    f16.missiles = ["AIM-120C"] * 6  # 6x AMRAAM

    # Su-27 (Red) - starting from east, flying west
    separation = 120000  # 120 km initial separation
    su27_pos = Vector3(separation, 0, altitude)
    su27_vel = Vector3(-speed, 0, 0)  # Flying west (toward F-16)
    su27 = Su27(su27_pos, su27_vel, name="Flanker-1")

    # Load missiles
    su27.missiles = ["R-77"] * 8  # 8x R-77 (more than MiG-29)

    # Print initial setup
    print("Initial Setup:")
    print("-" * 80)
    print(f"  F-16C: {f16}")
    print(f"    Position: {f16.position}")
    print(f"    Speed: {f16.get_speed():.0f} m/s ({f16.get_speed()/0.514444:.0f} kts)")
    print(f"    Missiles: {len(f16.missiles)}x {f16.missile_type}")
    print(f"    Radar: {f16.base_detection_range/1000:.0f} km vs 5m² target")
    print()
    print(f"  Su-27: {su27}")
    print(f"    Position: {su27.position}")
    print(f"    Speed: {su27.get_speed():.0f} m/s ({su27.get_speed()/0.514444:.0f} kts)")
    print(f"    Missiles: {len(su27.missiles)}x {su27.missile_type}")
    print(f"    Radar: {su27.base_detection_range/1000:.0f} km vs 5m² target")
    print()

    # Calculate detection ranges
    print("Detection Ranges:")
    print("-" * 80)
    f16_rcs = f16.get_rcs(su27.position)
    su27_rcs = su27.get_rcs(f16.position)

    f16_detect_range = f16.get_detection_range(su27_rcs)
    su27_detect_range = su27.get_detection_range(f16_rcs)

    print(f"  F-16 RCS (from Su-27): {f16_rcs:.1f} m²")
    print(f"  Su-27 RCS (from F-16): {su27_rcs:.1f} m²")
    print()
    print(f"  F-16 can detect Su-27 at: {f16_detect_range/1000:.1f} km")
    print(f"  Su-27 can detect F-16 at: {su27_detect_range/1000:.1f} km")
    print()

    advantage = f16_detect_range - su27_detect_range
    if advantage > 0:
        print(f"  Detection advantage: F-16 by {advantage/1000:.1f} km")
    else:
        print(f"  Detection advantage: Su-27 by {-advantage/1000:.1f} km")
    print()

    # Create engagement
    blue_force = [f16]
    red_force = [su27]

    engagement = Engagement(blue_force, red_force, dt=1.0)

    # Run engagement with progress updates
    print("Engagement in progress...")
    print("-" * 80)
    last_print = 0
    for i in range(600):
        engagement.step()
        range_between = f16.position.distance_to(su27.position)

        # Print every 20 seconds or when something interesting happens
        if engagement.time - last_print >= 20 or len(engagement.missiles) > 0:
            f16_can_see = f16.can_detect(su27)
            su27_can_see = su27.can_detect(f16)
            print(f"t={engagement.time:5.0f}s  Range:{range_between/1000:6.1f}km  "
                  f"F16 sees:{f16_can_see}  Su27 sees:{su27_can_see}  Missiles:{len(engagement.missiles)}")
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
        print("F-16 victory!")
        print("\nKey factors:")
        print(f"  - Detection range advantage: {advantage/1000:.1f} km")
        print("  - Lower RCS (5m² vs 15m²) = harder to detect")
        print("  - Slightly longer-range missiles (AIM-120C: 105km vs R-77: 80km)")
        print("  - Shot first, maintained BVR advantage")

    elif results['red_survivors'] > 0 and results['blue_survivors'] == 0:
        print("Su-27 victory!")
        print("\nKey factors:")
        print("  - Better radar range than MiG-29 (110km vs 75km)")
        print("  - More missiles available (8 vs 6)")
        print("  - Managed to get first effective shot despite disadvantages")

    else:
        print("Mutual kill or draw")
        print("\nBoth aircraft have significant advantages:")
        print("  F-16: Lower RCS, better radar, longer-range missiles")
        print("  Su-27: More missiles, similar performance")

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

    # Try to display, or save to files if no display available
    try:
        # Check if we can display interactively
        backend = plt.get_backend()
        if 'agg' in backend.lower():
            raise RuntimeError("Non-interactive backend")

        print("Displaying plots... (close windows to exit)")
        plt.show()
    except (RuntimeError, Exception):
        # Save to files instead
        print("No display available - saving plots to files...")

        output_dir = os.path.join(parent_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)

        engagement_file = os.path.join(output_dir, 'f16_vs_su27_engagement.png')
        timeline_file = os.path.join(output_dir, 'f16_vs_su27_timeline.png')

        fig1.savefig(engagement_file, dpi=150, bbox_inches='tight')
        fig2.savefig(timeline_file, dpi=150, bbox_inches='tight')

        print(f"  Engagement plot saved to: {engagement_file}")
        print(f"  Timeline plot saved to: {timeline_file}")
        print()

        plt.close('all')


if __name__ == "__main__":
    main()
