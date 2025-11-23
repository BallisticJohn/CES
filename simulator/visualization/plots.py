"""
Visualization tools for air combat simulation
Includes trajectory plots, energy-maneuverability diagrams, and analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..physics.atmosphere import Atmosphere


class TrajectoryPlotter:
    """Plot aircraft trajectories in 3D"""

    @staticmethod
    def plot_3d_trajectory(engagement_sim, interactive: bool = False,
                          show_velocity_vectors: bool = False):
        """
        Plot 3D trajectories from engagement simulation

        Args:
            engagement_sim: EngagementSimulation object with history
            interactive: If True, use plotly for interactive plot
            show_velocity_vectors: Show velocity vectors along path
        """
        if not engagement_sim.history['time']:
            print("No history to plot - run simulation first")
            return

        times = np.array(engagement_sim.history['time'])
        pos1 = np.array(engagement_sim.history['aircraft1']['position'])
        pos2 = np.array(engagement_sim.history['aircraft2']['position'])
        vel1 = np.array(engagement_sim.history['aircraft1']['velocity'])
        vel2 = np.array(engagement_sim.history['aircraft2']['velocity'])

        # Convert Z to altitude (positive up)
        alt1 = -pos1[:, 2]
        alt2 = -pos2[:, 2]

        if interactive:
            TrajectoryPlotter._plot_3d_interactive(
                times, pos1, pos2, alt1, alt2,
                engagement_sim.aircraft1, engagement_sim.aircraft2
            )
        else:
            TrajectoryPlotter._plot_3d_matplotlib(
                times, pos1, pos2, alt1, alt2, vel1, vel2,
                engagement_sim.aircraft1, engagement_sim.aircraft2,
                show_velocity_vectors
            )

    @staticmethod
    def _plot_3d_matplotlib(times, pos1, pos2, alt1, alt2, vel1, vel2,
                           aircraft1, aircraft2, show_velocity_vectors):
        """Create 3D plot using matplotlib"""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig)

        # 3D trajectory
        ax3d = fig.add_subplot(gs[:, 0], projection='3d')

        # Plot paths
        ax3d.plot(pos1[:, 0]/1000, pos1[:, 1]/1000, alt1/1000,
                 label=f'{type(aircraft1).__name__}', linewidth=2, color='blue')
        ax3d.plot(pos2[:, 0]/1000, pos2[:, 1]/1000, alt2/1000,
                 label=f'{type(aircraft2).__name__}', linewidth=2, color='red')

        # Mark start and end
        ax3d.scatter([pos1[0, 0]/1000], [pos1[0, 1]/1000], [alt1[0]/1000],
                    c='green', s=150, marker='o', label='Start', zorder=5)
        ax3d.scatter([pos1[-1, 0]/1000], [pos1[-1, 1]/1000], [alt1[-1]/1000],
                    c='darkblue', s=150, marker='x', zorder=5)
        ax3d.scatter([pos2[-1, 0]/1000], [pos2[-1, 1]/1000], [alt2[-1]/1000],
                    c='darkred', s=150, marker='x', zorder=5)

        # Velocity vectors (every 20th point)
        if show_velocity_vectors:
            step = max(1, len(times) // 20)
            for i in range(0, len(times), step):
                v1_norm = vel1[i] / np.linalg.norm(vel1[i]) * 2  # 2km arrows
                v2_norm = vel2[i] / np.linalg.norm(vel2[i]) * 2
                ax3d.quiver(pos1[i, 0]/1000, pos1[i, 1]/1000, alt1[i]/1000,
                          v1_norm[0], v1_norm[1], -v1_norm[2],
                          color='blue', alpha=0.3, arrow_length_ratio=0.3)
                ax3d.quiver(pos2[i, 0]/1000, pos2[i, 1]/1000, alt2[i]/1000,
                          v2_norm[0], v2_norm[1], -v2_norm[2],
                          color='red', alpha=0.3, arrow_length_ratio=0.3)

        ax3d.set_xlabel('North (km)')
        ax3d.set_ylabel('East (km)')
        ax3d.set_zlabel('Altitude (km)')
        ax3d.set_title('3D Engagement Trajectory', fontsize=14, fontweight='bold')
        ax3d.legend()
        ax3d.grid(True, alpha=0.3)

        # Top-down view
        ax_top = fig.add_subplot(gs[0, 1])
        ax_top.plot(pos1[:, 0]/1000, pos1[:, 1]/1000, 'b-', linewidth=2,
                   label=type(aircraft1).__name__)
        ax_top.plot(pos2[:, 0]/1000, pos2[:, 1]/1000, 'r-', linewidth=2,
                   label=type(aircraft2).__name__)
        ax_top.scatter([pos1[0, 0]/1000], [pos1[0, 1]/1000], c='green', s=100, marker='o', zorder=5)
        ax_top.scatter([pos1[-1, 0]/1000], [pos1[-1, 1]/1000], c='darkblue', s=100, marker='x', zorder=5)
        ax_top.scatter([pos2[-1, 0]/1000], [pos2[-1, 1]/1000], c='darkred', s=100, marker='x', zorder=5)
        ax_top.set_xlabel('North (km)')
        ax_top.set_ylabel('East (km)')
        ax_top.set_title('Top-Down View')
        ax_top.grid(True, alpha=0.3)
        ax_top.legend()
        ax_top.axis('equal')

        # Altitude vs time
        ax_alt = fig.add_subplot(gs[1, 1])
        ax_alt.plot(times, Atmosphere.meters_to_feet(alt1), 'b-', linewidth=2,
                   label=type(aircraft1).__name__)
        ax_alt.plot(times, Atmosphere.meters_to_feet(alt2), 'r-', linewidth=2,
                   label=type(aircraft2).__name__)
        ax_alt.set_xlabel('Time (s)')
        ax_alt.set_ylabel('Altitude (ft)')
        ax_alt.set_title('Altitude History')
        ax_alt.grid(True, alpha=0.3)
        ax_alt.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _plot_3d_interactive(times, pos1, pos2, alt1, alt2, aircraft1, aircraft2):
        """Create interactive 3D plot using plotly"""
        fig = go.Figure()

        # Aircraft 1 trajectory
        fig.add_trace(go.Scatter3d(
            x=pos1[:, 0]/1000,
            y=pos1[:, 1]/1000,
            z=alt1/1000,
            mode='lines',
            name=type(aircraft1).__name__,
            line=dict(color='blue', width=4),
            hovertemplate='Time: %{text}s<br>N: %{x:.2f} km<br>E: %{y:.2f} km<br>Alt: %{z:.2f} km',
            text=[f'{t:.1f}' for t in times]
        ))

        # Aircraft 2 trajectory
        fig.add_trace(go.Scatter3d(
            x=pos2[:, 0]/1000,
            y=pos2[:, 1]/1000,
            z=alt2/1000,
            mode='lines',
            name=type(aircraft2).__name__,
            line=dict(color='red', width=4),
            hovertemplate='Time: %{text}s<br>N: %{x:.2f} km<br>E: %{y:.2f} km<br>Alt: %{z:.2f} km',
            text=[f'{t:.1f}' for t in times]
        ))

        # Start/end markers
        fig.add_trace(go.Scatter3d(
            x=[pos1[0, 0]/1000],
            y=[pos1[0, 1]/1000],
            z=[alt1[0]/1000],
            mode='markers',
            name='Start',
            marker=dict(size=10, color='green', symbol='circle')
        ))

        fig.update_layout(
            title='Interactive 3D Engagement',
            scene=dict(
                xaxis_title='North (km)',
                yaxis_title='East (km)',
                zaxis_title='Altitude (km)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            hovermode='closest'
        )

        fig.show()


class PerformancePlotter:
    """Plot aircraft performance diagrams"""

    @staticmethod
    def plot_energy_maneuverability(aircraft, altitudes: Optional[List[float]] = None,
                                   velocities: Optional[List[float]] = None):
        """
        Plot energy-maneuverability (E-M) diagrams showing Ps contours

        Args:
            aircraft: Aircraft instance
            altitudes: List of altitudes to plot (m), defaults to several levels
            velocities: Velocity range (m/s), defaults to 100-350 m/s
        """
        if altitudes is None:
            altitudes = [
                Atmosphere.feet_to_meters(10000),
                Atmosphere.feet_to_meters(20000),
                Atmosphere.feet_to_meters(30000),
                Atmosphere.feet_to_meters(40000),
            ]

        if velocities is None:
            velocities = np.linspace(100, 350, 50)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{type(aircraft).__name__} Energy-Maneuverability Diagram',
                    fontsize=16, fontweight='bold')

        for idx, altitude in enumerate(altitudes):
            ax = axes[idx // 2, idx % 2]

            # Calculate Ps for each velocity
            ps_values = []
            for v in velocities:
                try:
                    ps = aircraft.get_specific_excess_power(altitude, v)
                    ps_values.append(ps)
                except:
                    ps_values.append(np.nan)

            ps_values = np.array(ps_values)

            # Convert to ft/min for display
            ps_fpm = ps_values / 0.3048 * 60

            # Plot Ps curve
            ax.plot(Atmosphere.mps_to_knots(velocities), ps_fpm,
                   linewidth=2, color='blue')
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.grid(True, alpha=0.3)

            # Labels
            ax.set_xlabel('Airspeed (knots)')
            ax.set_ylabel('Specific Excess Power (ft/min)')
            ax.set_title(f'Altitude: {Atmosphere.meters_to_feet(altitude):.0f} ft')

            # Add corner velocity marker
            corner_v = aircraft.get_corner_velocity(altitude)
            if 100 <= corner_v <= 350:
                corner_ps = aircraft.get_specific_excess_power(altitude, corner_v)
                ax.plot(Atmosphere.mps_to_knots(corner_v),
                       corner_ps / 0.3048 * 60,
                       'ro', markersize=8, label='Corner velocity')
                ax.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_turn_performance(aircraft, altitudes: Optional[List[float]] = None):
        """
        Plot turn performance (sustained turn rate vs velocity)

        Args:
            aircraft: Aircraft instance
            altitudes: Altitudes to plot (m)
        """
        if altitudes is None:
            altitudes = [
                Atmosphere.feet_to_meters(10000),
                Atmosphere.feet_to_meters(20000),
                Atmosphere.feet_to_meters(30000),
            ]

        velocities = np.linspace(100, 350, 40)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{type(aircraft).__name__} Turn Performance',
                    fontsize=16, fontweight='bold')

        colors = ['blue', 'green', 'red']

        for altitude, color in zip(altitudes, colors):
            turn_rates = []
            turn_radius = []

            for v in velocities:
                tr = aircraft.get_sustained_turn_rate(altitude, v)
                turn_rates.append(np.degrees(tr))

                # Turn radius = V / omega
                if tr > 0:
                    radius = v / tr
                    turn_radius.append(radius)
                else:
                    turn_radius.append(np.nan)

            alt_ft = Atmosphere.meters_to_feet(altitude)

            # Turn rate plot
            ax1.plot(Atmosphere.mps_to_knots(velocities), turn_rates,
                    linewidth=2, color=color, label=f'{alt_ft:.0f} ft')

            # Turn radius plot
            ax2.plot(Atmosphere.mps_to_knots(velocities),
                    np.array(turn_radius) / 1000,  # km
                    linewidth=2, color=color, label=f'{alt_ft:.0f} ft')

        ax1.set_xlabel('Airspeed (knots)')
        ax1.set_ylabel('Sustained Turn Rate (deg/s)')
        ax1.set_title('Turn Rate vs Speed')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.set_xlabel('Airspeed (knots)')
        ax2.set_ylabel('Turn Radius (km)')
        ax2.set_title('Turn Radius vs Speed')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_comparative_performance(aircraft1, aircraft2,
                                    altitude: float = None):
        """
        Compare performance of two aircraft

        Args:
            aircraft1: First aircraft
            aircraft2: Second aircraft
            altitude: Altitude for comparison (m)
        """
        if altitude is None:
            altitude = Atmosphere.feet_to_meters(25000)

        velocities = np.linspace(100, 350, 50)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Performance Comparison at {Atmosphere.meters_to_feet(altitude):.0f} ft',
                    fontsize=16, fontweight='bold')

        # Specific excess power
        ax = axes[0, 0]
        ps1 = [aircraft1.get_specific_excess_power(altitude, v) for v in velocities]
        ps2 = [aircraft2.get_specific_excess_power(altitude, v) for v in velocities]
        ax.plot(Atmosphere.mps_to_knots(velocities),
               np.array(ps1) / 0.3048 * 60,
               linewidth=2, label=type(aircraft1).__name__, color='blue')
        ax.plot(Atmosphere.mps_to_knots(velocities),
               np.array(ps2) / 0.3048 * 60,
               linewidth=2, label=type(aircraft2).__name__, color='red')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Airspeed (knots)')
        ax.set_ylabel('Ps (ft/min)')
        ax.set_title('Specific Excess Power')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Turn rate
        ax = axes[0, 1]
        tr1 = [np.degrees(aircraft1.get_sustained_turn_rate(altitude, v)) for v in velocities]
        tr2 = [np.degrees(aircraft2.get_sustained_turn_rate(altitude, v)) for v in velocities]
        ax.plot(Atmosphere.mps_to_knots(velocities), tr1,
               linewidth=2, label=type(aircraft1).__name__, color='blue')
        ax.plot(Atmosphere.mps_to_knots(velocities), tr2,
               linewidth=2, label=type(aircraft2).__name__, color='red')
        ax.set_xlabel('Airspeed (knots)')
        ax.set_ylabel('Turn Rate (deg/s)')
        ax.set_title('Sustained Turn Rate')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Thrust comparison
        ax = axes[1, 0]
        thrust1 = [aircraft1.get_thrust(1.0, altitude, Atmosphere.get_mach_number(v, altitude))
                  for v in velocities]
        thrust2 = [aircraft2.get_thrust(1.0, altitude, Atmosphere.get_mach_number(v, altitude))
                  for v in velocities]
        ax.plot(Atmosphere.mps_to_knots(velocities),
               np.array(thrust1) / 1000,
               linewidth=2, label=type(aircraft1).__name__, color='blue')
        ax.plot(Atmosphere.mps_to_knots(velocities),
               np.array(thrust2) / 1000,
               linewidth=2, label=type(aircraft2).__name__, color='red')
        ax.set_xlabel('Airspeed (knots)')
        ax.set_ylabel('Thrust (kN)')
        ax.set_title('Available Thrust')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Corner velocity comparison
        ax = axes[1, 1]
        altitudes = np.linspace(Atmosphere.feet_to_meters(5000),
                               Atmosphere.feet_to_meters(45000), 20)
        corner1 = [aircraft1.get_corner_velocity(alt) for alt in altitudes]
        corner2 = [aircraft2.get_corner_velocity(alt) for alt in altitudes]
        ax.plot(Atmosphere.mps_to_knots(corner1),
               Atmosphere.meters_to_feet(altitudes) / 1000,
               linewidth=2, label=type(aircraft1).__name__, color='blue')
        ax.plot(Atmosphere.mps_to_knots(corner2),
               Atmosphere.meters_to_feet(altitudes) / 1000,
               linewidth=2, label=type(aircraft2).__name__, color='red')
        ax.set_xlabel('Corner Velocity (knots)')
        ax.set_ylabel('Altitude (1000 ft)')
        ax.set_title('Corner Velocity vs Altitude')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()


class EngagementAnalyzer:
    """Analyze engagement results"""

    @staticmethod
    def plot_engagement_analysis(engagement_sim):
        """
        Comprehensive engagement analysis plots

        Args:
            engagement_sim: EngagementSimulation with history
        """
        if not engagement_sim.history['time']:
            print("No history to analyze")
            return

        times = np.array(engagement_sim.history['time'])
        pos1 = np.array(engagement_sim.history['aircraft1']['position'])
        pos2 = np.array(engagement_sim.history['aircraft2']['position'])
        vel1 = np.array(engagement_sim.history['aircraft1']['velocity'])
        vel2 = np.array(engagement_sim.history['aircraft2']['velocity'])
        alt1 = np.array(engagement_sim.history['aircraft1']['altitude'])
        alt2 = np.array(engagement_sim.history['aircraft2']['altitude'])

        # Calculate metrics
        separations = [np.linalg.norm(pos1[i] - pos2[i]) for i in range(len(times))]
        speeds1 = [np.linalg.norm(vel1[i]) for i in range(len(times))]
        speeds2 = [np.linalg.norm(vel2[i]) for i in range(len(times))]

        # Aspect angles
        aspects1 = []
        aspects2 = []
        for i in range(len(times)):
            asp1, asp2 = engagement_sim.get_aspect_angle(i)
            aspects1.append(asp1)
            aspects2.append(asp2)

        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 2, figure=fig)

        # Separation distance
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(times, np.array(separations) / 1000, 'k-', linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Separation (km)')
        ax1.set_title('Aircraft Separation', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        min_sep_idx = np.argmin(separations)
        ax1.plot(times[min_sep_idx], separations[min_sep_idx] / 1000,
                'ro', markersize=10, label=f'Min: {separations[min_sep_idx]:.0f}m')
        ax1.legend()

        # Airspeeds
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(times, Atmosphere.mps_to_knots(speeds1),
                'b-', linewidth=2, label=type(engagement_sim.aircraft1).__name__)
        ax2.plot(times, Atmosphere.mps_to_knots(speeds2),
                'r-', linewidth=2, label=type(engagement_sim.aircraft2).__name__)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Airspeed (knots)')
        ax2.set_title('Airspeed History', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Altitudes
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(times, Atmosphere.meters_to_feet(alt1),
                'b-', linewidth=2, label=type(engagement_sim.aircraft1).__name__)
        ax3.plot(times, Atmosphere.meters_to_feet(alt2),
                'r-', linewidth=2, label=type(engagement_sim.aircraft2).__name__)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Altitude (ft)')
        ax3.set_title('Altitude History', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Aspect angles
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(times, aspects1, 'b-', linewidth=2,
                label=f'{type(engagement_sim.aircraft1).__name__} aspect')
        ax4.plot(times, aspects2, 'r-', linewidth=2,
                label=f'{type(engagement_sim.aircraft2).__name__} aspect')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Aspect Angle (deg)')
        ax4.set_title('Aspect Angles (0°=tail, 180°=head-on)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        # Energy state (specific energy = altitude + V²/2g)
        ax5 = fig.add_subplot(gs[2, :])
        E1 = alt1 + np.array(speeds1)**2 / (2 * Atmosphere.G)
        E2 = alt2 + np.array(speeds2)**2 / (2 * Atmosphere.G)
        ax5.plot(times, Atmosphere.meters_to_feet(E1),
                'b-', linewidth=2, label=type(engagement_sim.aircraft1).__name__)
        ax5.plot(times, Atmosphere.meters_to_feet(E2),
                'r-', linewidth=2, label=type(engagement_sim.aircraft2).__name__)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Specific Energy (ft)')
        ax5.set_title('Energy State (Altitude + Kinetic Energy)', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend()

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        print("\n" + "="*60)
        print("ENGAGEMENT ANALYSIS SUMMARY")
        print("="*60)
        print(f"Duration: {times[-1]:.1f} seconds")
        print(f"Minimum separation: {min(separations):.0f} m at t={times[np.argmin(separations)]:.1f}s")
        print(f"\n{type(engagement_sim.aircraft1).__name__}:")
        print(f"  Altitude change: {Atmosphere.meters_to_feet(alt1[-1] - alt1[0]):+.0f} ft")
        print(f"  Speed change: {Atmosphere.mps_to_knots(speeds1[-1] - speeds1[0]):+.0f} knots")
        print(f"  Energy change: {Atmosphere.meters_to_feet(E1[-1] - E1[0]):+.0f} ft")
        print(f"\n{type(engagement_sim.aircraft2).__name__}:")
        print(f"  Altitude change: {Atmosphere.meters_to_feet(alt2[-1] - alt2[0]):+.0f} ft")
        print(f"  Speed change: {Atmosphere.mps_to_knots(speeds2[-1] - speeds2[0]):+.0f} knots")
        print(f"  Energy change: {Atmosphere.meters_to_feet(E2[-1] - E2[0]):+.0f} ft")
        print("="*60 + "\n")
