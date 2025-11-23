"""
Engagement simulation between two aircraft
Handles combat scenarios and interaction
"""

import numpy as np
from typing import List, Tuple, TYPE_CHECKING
from ..physics.vector import Vector3

if TYPE_CHECKING:
    from ..aircraft.base import Aircraft


class EngagementSimulation:
    """
    Simulates air combat engagement between two aircraft

    Tracks both aircraft through a dogfight scenario
    """

    def __init__(self, aircraft1: 'Aircraft', aircraft2: 'Aircraft'):
        """
        Initialize engagement

        Args:
            aircraft1: First aircraft
            aircraft2: Second aircraft
        """
        self.aircraft1 = aircraft1
        self.aircraft2 = aircraft2

        # Simulation parameters
        self.dt = 0.05  # 50ms time step
        self.max_time = 300.0  # 5 minutes max

        # History tracking
        self.history = {
            'time': [],
            'aircraft1': {'position': [], 'velocity': [], 'altitude': []},
            'aircraft2': {'position': [], 'velocity': [], 'altitude': []},
        }

        # Engagement results
        self.result = None

    def run(self, max_time: float = None, dt: float = None) -> dict:
        """
        Run the engagement simulation

        Args:
            max_time: Maximum simulation time (s)
            dt: Time step (s)

        Returns:
            Results dictionary
        """
        if max_time is not None:
            self.max_time = max_time
        if dt is not None:
            self.dt = dt

        time = 0.0
        self.history['time'] = []

        while time < self.max_time:
            # Record state
            self._record_state(time)

            # Update both aircraft
            self.aircraft1.update(self.dt)
            self.aircraft2.update(self.dt)

            # Check termination conditions
            if self._check_termination():
                break

            time += self.dt

        # Compile results
        self.result = self._compile_results()
        return self.result

    def _record_state(self, time: float):
        """Record current state of both aircraft"""
        self.history['time'].append(time)

        # Aircraft 1
        self.history['aircraft1']['position'].append(
            self.aircraft1.state.position.to_array()
        )
        self.history['aircraft1']['velocity'].append(
            self.aircraft1.state.velocity.to_array()
        )
        self.history['aircraft1']['altitude'].append(
            self.aircraft1.get_altitude()
        )

        # Aircraft 2
        self.history['aircraft2']['position'].append(
            self.aircraft2.state.position.to_array()
        )
        self.history['aircraft2']['velocity'].append(
            self.aircraft2.state.velocity.to_array()
        )
        self.history['aircraft2']['altitude'].append(
            self.aircraft2.get_altitude()
        )

    def _check_termination(self) -> bool:
        """Check if simulation should terminate"""
        # Check if either aircraft crashed
        if self.aircraft1.get_altitude() < 0:
            self.result = {'winner': 'aircraft2', 'reason': 'aircraft1_crashed'}
            return True
        if self.aircraft2.get_altitude() < 0:
            self.result = {'winner': 'aircraft1', 'reason': 'aircraft2_crashed'}
            return True

        # Check if destroyed
        if self.aircraft1.is_destroyed:
            self.result = {'winner': 'aircraft2', 'reason': 'aircraft1_destroyed'}
            return True
        if self.aircraft2.is_destroyed:
            self.result = {'winner': 'aircraft1', 'reason': 'aircraft2_destroyed'}
            return True

        return False

    def _compile_results(self) -> dict:
        """Compile final results"""
        if self.result is None:
            self.result = {'winner': None, 'reason': 'time_limit'}

        # Add statistics
        self.result['duration'] = self.history['time'][-1] if self.history['time'] else 0
        self.result['final_separation'] = self.get_separation(-1)

        return self.result

    def get_separation(self, index: int = -1) -> float:
        """
        Get separation distance between aircraft

        Args:
            index: History index (-1 for latest)

        Returns:
            Distance in meters
        """
        if not self.history['time']:
            pos1 = self.aircraft1.state.position
            pos2 = self.aircraft2.state.position
        else:
            pos1 = Vector3.from_array(self.history['aircraft1']['position'][index])
            pos2 = Vector3.from_array(self.history['aircraft2']['position'][index])

        return pos1.distance_to(pos2)

    def get_aspect_angle(self, index: int = -1) -> Tuple[float, float]:
        """
        Get aspect angles (how each aircraft sees the other)

        Args:
            index: History index

        Returns:
            (aspect1_to_2, aspect2_to_1) in degrees
            0° = tail aspect, 180° = head-on
        """
        if not self.history['time']:
            return 0.0, 0.0

        # Get positions and velocities
        pos1 = Vector3.from_array(self.history['aircraft1']['position'][index])
        pos2 = Vector3.from_array(self.history['aircraft2']['position'][index])
        vel1 = Vector3.from_array(self.history['aircraft1']['velocity'][index])
        vel2 = Vector3.from_array(self.history['aircraft2']['velocity'][index])

        # Vector from 1 to 2
        to_2 = (pos2 - pos1).normalized()
        # Aspect: angle between aircraft1's velocity and direction to aircraft2
        aspect1 = np.degrees(vel1.normalized().angle_to(to_2))

        # Vector from 2 to 1
        to_1 = (pos1 - pos2).normalized()
        aspect2 = np.degrees(vel2.normalized().angle_to(to_1))

        return aspect1, aspect2

    def plot_trajectories(self):
        """Plot 3D trajectories (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Convert to arrays
            pos1 = np.array(self.history['aircraft1']['position'])
            pos2 = np.array(self.history['aircraft2']['position'])

            # Plot trajectories
            ax.plot(pos1[:, 0], pos1[:, 1], -pos1[:, 2],
                   label=f'{type(self.aircraft1).__name__}', linewidth=2)
            ax.plot(pos2[:, 0], pos2[:, 1], -pos2[:, 2],
                   label=f'{type(self.aircraft2).__name__}', linewidth=2)

            # Mark start and end
            ax.scatter([pos1[0, 0]], [pos1[0, 1]], [-pos1[0, 2]],
                      c='green', s=100, marker='o', label='Start')
            ax.scatter([pos1[-1, 0]], [pos1[-1, 1]], [-pos1[-1, 2]],
                      c='red', s=100, marker='x', label='End')

            ax.set_xlabel('North (m)')
            ax.set_ylabel('East (m)')
            ax.set_zlabel('Altitude (m)')
            ax.set_title('Engagement Trajectories')
            ax.legend()

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Matplotlib not available for plotting")

    def __repr__(self) -> str:
        """String representation"""
        if self.result:
            return (f"Engagement: {type(self.aircraft1).__name__} vs "
                   f"{type(self.aircraft2).__name__}\n"
                   f"Result: {self.result}")
        else:
            return (f"Engagement: {type(self.aircraft1).__name__} vs "
                   f"{type(self.aircraft2).__name__} (not yet run)")
