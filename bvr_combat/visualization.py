"""
Visualization tools for BVR engagements
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict
import numpy as np


def plot_engagement(blue_force, red_force, events: List[Dict],
                   blue_history: List, red_history: List,
                   missile_history: List):
    """
    Create a 2D visualization of the engagement

    Args:
        blue_force: List of blue aircraft
        red_force: List of red aircraft
        events: List of engagement events
        blue_history: Position history for blue force
        red_history: Position history for red force
        missile_history: Missile trajectory history
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot aircraft trajectories
    for i, aircraft_hist in enumerate(blue_history):
        if len(aircraft_hist) > 0:
            x = [p[0] / 1000 for p in aircraft_hist]  # Convert to km
            y = [p[1] / 1000 for p in aircraft_hist]
            ax.plot(x, y, 'b-', linewidth=2, label=f'Blue-{i+1}' if i == 0 else '')
            # Mark start and end
            ax.plot(x[0], y[0], 'bo', markersize=10)
            if blue_force[i].alive:
                ax.plot(x[-1], y[-1], 'b^', markersize=12)
            else:
                ax.plot(x[-1], y[-1], 'bx', markersize=12, markeredgewidth=3)

    for i, aircraft_hist in enumerate(red_history):
        if len(aircraft_hist) > 0:
            x = [p[0] / 1000 for p in aircraft_hist]
            y = [p[1] / 1000 for p in aircraft_hist]
            ax.plot(x, y, 'r-', linewidth=2, label=f'Red-{i+1}' if i == 0 else '')
            # Mark start and end
            ax.plot(x[0], y[0], 'ro', markersize=10)
            if red_force[i].alive:
                ax.plot(x[-1], y[-1], 'r^', markersize=12)
            else:
                ax.plot(x[-1], y[-1], 'rx', markersize=12, markeredgewidth=3)

    # Plot missile trajectories
    for missile_traj in missile_history:
        if len(missile_traj) > 1:
            x = [p[0] / 1000 for p in missile_traj]
            y = [p[1] / 1000 for p in missile_traj]
            ax.plot(x, y, 'k--', linewidth=0.5, alpha=0.5)

    # Mark key events
    for event in events:
        if event['type'] == 'MISSILE_LAUNCH':
            # Find launch position from histories
            time_idx = int(event['time'])
            if 'BLUE' in event['shooter']:
                idx = int(event['shooter'].split('-')[1]) - 1
                if idx < len(blue_history) and time_idx < len(blue_history[idx]):
                    pos = blue_history[idx][time_idx]
                    ax.plot(pos[0]/1000, pos[1]/1000, 'g*', markersize=15,
                           markeredgecolor='black', markeredgewidth=0.5)

        elif event['type'] == 'KILL':
            # Mark kill location
            time_idx = int(event['time'])
            if 'BLUE' in event['victim']:
                idx = int(event['victim'].split('-')[1]) - 1
                if idx < len(blue_history) and time_idx < len(blue_history[idx]):
                    pos = blue_history[idx][time_idx]
                    ax.plot(pos[0]/1000, pos[1]/1000, 'r*', markersize=20,
                           markeredgecolor='black', markeredgewidth=1)
                    ax.text(pos[0]/1000, pos[1]/1000 + 5, f"KILL\nt={event['time']:.0f}s",
                           ha='center', fontsize=8, bbox=dict(boxstyle='round',
                           facecolor='red', alpha=0.7))
            else:
                idx = int(event['victim'].split('-')[1]) - 1
                if idx < len(red_history) and time_idx < len(red_history[idx]):
                    pos = red_history[idx][time_idx]
                    ax.plot(pos[0]/1000, pos[1]/1000, 'b*', markersize=20,
                           markeredgecolor='black', markeredgewidth=1)
                    ax.text(pos[0]/1000, pos[1]/1000 + 5, f"KILL\nt={event['time']:.0f}s",
                           ha='center', fontsize=8, bbox=dict(boxstyle='round',
                           facecolor='blue', alpha=0.7))

    # Formatting
    ax.set_xlabel('X Position (km)', fontsize=12)
    ax.set_ylabel('Y Position (km)', fontsize=12)
    ax.set_title('BVR Engagement - Top View', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color='b', linewidth=2, label='Blue Force'),
        plt.Line2D([0], [0], color='r', linewidth=2, label='Red Force'),
        plt.Line2D([0], [0], color='k', linewidth=0.5, linestyle='--', label='Missiles'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k',
                  markersize=10, label='Start'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='k',
                  markersize=10, label='Survived'),
        plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='k',
                  markersize=10, label='Killed'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='g',
                  markersize=12, label='Launch', markeredgecolor='black'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    return fig


def plot_timeline(events: List[Dict], duration: float):
    """
    Create a timeline visualization of engagement events

    Args:
        events: List of engagement events
        duration: Total engagement duration
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Filter relevant events
    detections = [e for e in events if e['type'] == 'DETECTION']
    launches = [e for e in events if e['type'] == 'MISSILE_LAUNCH']
    kills = [e for e in events if e['type'] == 'KILL']

    # Plot detections
    for det in detections:
        color = 'blue' if 'BLUE' in det['detector'] else 'red'
        ax.scatter(det['time'], 1, c=color, s=100, marker='o', alpha=0.5)

    # Plot launches
    for launch in launches:
        color = 'blue' if 'BLUE' in launch['shooter'] else 'red'
        ax.scatter(launch['time'], 2, c=color, s=200, marker='*',
                  edgecolors='black', linewidths=1)
        ax.text(launch['time'], 2.2, f"{launch['range_km']:.0f}km",
               ha='center', fontsize=8)

    # Plot kills
    for kill in kills:
        color = 'blue' if 'BLUE' in kill['killer'] else 'red'
        ax.scatter(kill['time'], 3, c=color, s=300, marker='X',
                  edgecolors='black', linewidths=2)
        ax.text(kill['time'], 3.3, kill['victim'], ha='center', fontsize=8)

    # Formatting
    ax.set_xlim(0, duration)
    ax.set_ylim(0.5, 3.5)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Detections', 'Launches', 'Kills'], fontsize=11)
    ax.set_title('Engagement Timeline', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    return fig
