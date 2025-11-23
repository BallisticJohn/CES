"""
Monte Carlo simulation framework
Run many engagements with varying initial conditions
"""

import numpy as np
from typing import List, Dict, Callable
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


class MonteCarloRunner:
    """
    Monte Carlo simulation runner for air combat scenarios

    Runs many engagements with randomized initial conditions
    to build statistical understanding of outcomes
    """

    def __init__(self, aircraft1_class, aircraft2_class,
                 num_simulations: int = 1000):
        """
        Initialize Monte Carlo runner

        Args:
            aircraft1_class: Class for first aircraft
            aircraft2_class: Class for second aircraft
            num_simulations: Number of simulations to run
        """
        self.aircraft1_class = aircraft1_class
        self.aircraft2_class = aircraft2_class
        self.num_simulations = num_simulations

        # Parameter ranges for randomization
        self.altitude_range = (3000, 12000)  # meters (10k-40k ft)
        self.speed_range = (200, 300)       # m/s
        self.separation_range = (1000, 5000)  # meters
        self.heading_range = (0, 2 * np.pi)  # radians

        # Results storage
        self.results = []

    def set_parameter_ranges(self, altitude_range=None, speed_range=None,
                            separation_range=None, heading_range=None):
        """Set ranges for randomized parameters"""
        if altitude_range is not None:
            self.altitude_range = altitude_range
        if speed_range is not None:
            self.speed_range = speed_range
        if separation_range is not None:
            self.separation_range = separation_range
        if heading_range is not None:
            self.heading_range = heading_range

    def _run_single_simulation(self, seed: int) -> dict:
        """
        Run a single engagement with randomized parameters

        Args:
            seed: Random seed for reproducibility

        Returns:
            Result dictionary
        """
        from ..physics.vector import Vector3
        from ..physics.quaternion import Quaternion
        from .engagement import EngagementSimulation

        np.random.seed(seed)

        # Randomize initial conditions
        altitude = np.random.uniform(*self.altitude_range)
        speed1 = np.random.uniform(*self.speed_range)
        speed2 = np.random.uniform(*self.speed_range)
        separation = np.random.uniform(*self.separation_range)
        heading1 = np.random.uniform(*self.heading_range)
        heading2 = np.random.uniform(*self.heading_range)

        # Create aircraft
        # Aircraft 1 at origin
        pos1 = Vector3(0, 0, -altitude)
        vel1 = Vector3(speed1 * np.cos(heading1), speed1 * np.sin(heading1), 0)
        orient1 = Quaternion.from_euler(0, 0, heading1)

        # Aircraft 2 at separation distance
        angle = np.random.uniform(0, 2 * np.pi)
        pos2 = Vector3(separation * np.cos(angle), separation * np.sin(angle), -altitude)
        vel2 = Vector3(speed2 * np.cos(heading2), speed2 * np.sin(heading2), 0)
        orient2 = Quaternion.from_euler(0, 0, heading2)

        aircraft1 = self.aircraft1_class(pos1, vel1, orient1)
        aircraft2 = self.aircraft2_class(pos2, vel2, orient2)

        # Run engagement
        sim = EngagementSimulation(aircraft1, aircraft2)
        result = sim.run(max_time=300.0)

        # Add initial conditions to result
        result['initial'] = {
            'altitude': altitude,
            'speed1': speed1,
            'speed2': speed2,
            'separation': separation,
            'heading1': heading1,
            'heading2': heading2,
        }

        return result

    def run_sequential(self) -> List[dict]:
        """Run all simulations sequentially"""
        self.results = []
        for i in range(self.num_simulations):
            if i % 100 == 0:
                print(f"Running simulation {i}/{self.num_simulations}...")
            result = self._run_single_simulation(i)
            self.results.append(result)

        return self.results

    def run_parallel(self, n_cores: int = None) -> List[dict]:
        """
        Run simulations in parallel

        Args:
            n_cores: Number of CPU cores to use (None = all available)

        Returns:
            List of results
        """
        if n_cores is None:
            n_cores = multiprocessing.cpu_count()

        print(f"Running {self.num_simulations} simulations on {n_cores} cores...")

        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            seeds = range(self.num_simulations)
            self.results = list(executor.map(self._run_single_simulation, seeds))

        return self.results

    def get_statistics(self) -> dict:
        """
        Compile statistics from all simulations

        Returns:
            Dictionary of statistical results
        """
        if not self.results:
            return {}

        # Win rates
        aircraft1_wins = sum(1 for r in self.results if r.get('winner') == 'aircraft1')
        aircraft2_wins = sum(1 for r in self.results if r.get('winner') == 'aircraft2')
        draws = sum(1 for r in self.results if r.get('winner') is None)

        # Average engagement duration
        durations = [r.get('duration', 0) for r in self.results]

        # Reasons for outcome
        reasons = {}
        for r in self.results:
            reason = r.get('reason', 'unknown')
            reasons[reason] = reasons.get(reason, 0) + 1

        stats = {
            'total_simulations': len(self.results),
            'aircraft1_wins': aircraft1_wins,
            'aircraft2_wins': aircraft2_wins,
            'draws': draws,
            'aircraft1_win_rate': aircraft1_wins / len(self.results),
            'aircraft2_win_rate': aircraft2_wins / len(self.results),
            'draw_rate': draws / len(self.results),
            'avg_duration': np.mean(durations),
            'std_duration': np.std(durations),
            'outcome_reasons': reasons,
        }

        return stats

    def plot_statistics(self):
        """Plot statistical results"""
        try:
            import matplotlib.pyplot as plt

            stats = self.get_statistics()

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Win rates
            ax = axes[0, 0]
            labels = ['Aircraft 1', 'Aircraft 2', 'Draw']
            sizes = [stats['aircraft1_wins'], stats['aircraft2_wins'], stats['draws']]
            ax.pie(sizes, labels=labels, autopct='%1.1f%%')
            ax.set_title('Engagement Outcomes')

            # Duration histogram
            ax = axes[0, 1]
            durations = [r.get('duration', 0) for r in self.results]
            ax.hist(durations, bins=30, edgecolor='black')
            ax.set_xlabel('Duration (s)')
            ax.set_ylabel('Frequency')
            ax.set_title('Engagement Duration Distribution')

            # Outcome reasons
            ax = axes[1, 0]
            reasons = stats['outcome_reasons']
            ax.bar(reasons.keys(), reasons.values())
            ax.set_xlabel('Outcome Reason')
            ax.set_ylabel('Count')
            ax.set_title('Outcome Reasons')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # Win rate vs initial conditions (example: altitude)
            ax = axes[1, 1]
            altitudes = [r['initial']['altitude'] for r in self.results]
            winners = [1 if r.get('winner') == 'aircraft1' else 0 for r in self.results]
            ax.scatter(altitudes, winners, alpha=0.3)
            ax.set_xlabel('Initial Altitude (m)')
            ax.set_ylabel('Aircraft 1 Win (1=yes, 0=no)')
            ax.set_title('Win Rate vs Altitude')

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Matplotlib not available for plotting")

    def export_csv(self, filename: str):
        """Export results to CSV file"""
        try:
            import pandas as pd

            # Flatten results for CSV
            data = []
            for r in self.results:
                row = {
                    'winner': r.get('winner'),
                    'reason': r.get('reason'),
                    'duration': r.get('duration', 0),
                    **r.get('initial', {})
                }
                data.append(row)

            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            print(f"Results exported to {filename}")

        except ImportError:
            print("Pandas not available for CSV export")
