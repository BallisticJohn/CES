"""
BVR engagement framework
Manages detection, missile launches, and battle outcomes
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from ..aircraft.base import Aircraft
from ..weapons.missile import Missile, AIM120C, R77
from ..utils import Vector3, meters_to_km


class Engagement:
    """
    Manages a BVR air combat engagement between two forces
    """

    def __init__(self, blue_force: List[Aircraft], red_force: List[Aircraft],
                 dt: float = 1.0):
        """
        Initialize engagement

        Args:
            blue_force: List of blue team aircraft
            red_force: List of red team aircraft
            dt: Simulation timestep in seconds
        """
        self.blue_force = blue_force
        self.red_force = red_force
        self.dt = dt

        # Assign unique IDs
        for i, aircraft in enumerate(blue_force):
            aircraft.id = f"BLUE-{i+1}"
        for i, aircraft in enumerate(red_force):
            aircraft.id = f"RED-{i+1}"

        # Active missiles
        self.missiles: List[Missile] = []

        # Engagement state
        self.time = 0.0
        self.max_time = 600.0  # 10 minute max engagement
        self.engagement_active = True

        # Event log
        self.events: List[Dict] = []

        # Statistics
        self.blue_kills = 0
        self.red_kills = 0

    def log_event(self, event_type: str, **kwargs):
        """Log an engagement event"""
        event = {
            'time': self.time,
            'type': event_type,
            **kwargs
        }
        self.events.append(event)

    def update_detections(self):
        """Update which aircraft can detect which targets"""
        all_aircraft = self.blue_force + self.red_force

        for aircraft in all_aircraft:
            if not aircraft.alive:
                continue

            # Determine friendly and enemy forces
            if aircraft in self.blue_force:
                enemies = self.red_force
            else:
                enemies = self.blue_force

            # Check detection of each enemy
            for enemy in enemies:
                if not enemy.alive:
                    continue

                can_detect = aircraft.can_detect(enemy)

                if can_detect and enemy.id not in aircraft.detected_targets:
                    # New detection
                    aircraft.detected_targets[enemy.id] = self.time
                    range_km = meters_to_km(aircraft.position.distance_to(enemy.position))
                    self.log_event('DETECTION',
                                 detector=aircraft.id,
                                 target=enemy.id,
                                 range_km=range_km)

    def make_launch_decision(self, shooter: Aircraft, target: Aircraft) -> bool:
        """
        Decide whether to launch a missile

        Simple decision logic:
        - Target detected and within range
        - Have missiles remaining
        - Not already launching at this target
        - Within reasonable launch parameters

        Args:
            shooter: Shooting aircraft
            target: Target aircraft

        Returns:
            True if should launch
        """
        if not shooter.alive or not target.alive:
            return False

        if len(shooter.missiles) == 0:
            return False

        # Check if target is detected
        if target.id not in shooter.detected_targets:
            return False

        # Calculate range and geometry
        range_m = shooter.position.distance_to(target.position)

        # Don't launch if too close (WEZ minimum) or too far
        min_launch_range = 5000  # 5 km minimum
        max_launch_range = 80000  # 80 km maximum (conservative)

        if range_m < min_launch_range or range_m > max_launch_range:
            return False

        # Check if we already have missiles in flight at this target
        missiles_at_target = sum(1 for m in self.missiles
                                if m.target_id == target.id and m.active)

        # Limit missiles per target (don't waste them)
        if missiles_at_target >= 2:
            return False

        # Simple launch decision: launch if in good range
        optimal_range_min = 20000  # 20 km
        optimal_range_max = 60000  # 60 km

        if optimal_range_min <= range_m <= optimal_range_max:
            return True

        return False

    def launch_missile(self, shooter: Aircraft, target: Aircraft):
        """
        Launch a missile from shooter at target

        Args:
            shooter: Shooting aircraft
            target: Target aircraft
        """
        if len(shooter.missiles) == 0:
            return

        # Remove missile from aircraft inventory
        missile_type = shooter.missiles.pop(0)

        # Create missile with aircraft velocity as initial velocity
        launch_pos = Vector3(shooter.position.x, shooter.position.y,
                            shooter.position.z)
        launch_vel = Vector3(shooter.velocity.x, shooter.velocity.y,
                            shooter.velocity.z)

        # Determine missile type
        if missile_type == "AIM-120C":
            missile = AIM120C(launch_pos, launch_vel, target.id)
        elif missile_type == "R-77":
            missile = R77(launch_pos, launch_vel, target.id)
        else:
            missile = AIM120C(launch_pos, launch_vel, target.id)  # Default

        self.missiles.append(missile)

        # Log launch
        range_km = meters_to_km(shooter.position.distance_to(target.position))
        self.log_event('MISSILE_LAUNCH',
                     shooter=shooter.id,
                     target=target.id,
                     missile=missile.name,
                     range_km=range_km)

    def update_missiles(self):
        """Update all active missiles and check for hits"""
        all_aircraft = self.blue_force + self.red_force

        for missile in self.missiles[:]:  # Copy list to allow removal
            if not missile.is_active():
                self.missiles.remove(missile)
                continue

            # Find target aircraft
            target = None
            for aircraft in all_aircraft:
                if aircraft.id == missile.target_id:
                    target = aircraft
                    break

            if target is None or not target.alive:
                missile.active = False
                continue

            # Update missile guidance
            missile.update(self.dt, target.position, target.velocity)

            # Check for proximity kill
            pk = missile.check_proximity(target.position, target.get_speed())

            if pk is not None:
                # Missile within lethal radius
                kill = np.random.random() < pk

                if kill:
                    target.kill()

                    # Determine which side scored kill
                    if target in self.blue_force:
                        self.red_kills += 1
                        self.log_event('KILL',
                                     victim=target.id,
                                     killer='RED (missile)',
                                     pk=pk)
                    else:
                        self.blue_kills += 1
                        self.log_event('KILL',
                                     victim=target.id,
                                     killer='BLUE (missile)',
                                     pk=pk)

                # Missile detonates regardless of hit/miss
                missile.active = False
                self.missiles.remove(missile)

    def simple_ai_control(self, aircraft: Aircraft, enemies: List[Aircraft]):
        """
        Simple AI control for aircraft

        Basic tactics:
        - If enemies detected, turn toward nearest and maintain speed
        - If no enemies, cruise forward

        Args:
            aircraft: Aircraft to control
            enemies: List of enemy aircraft
        """
        if not aircraft.alive:
            return

        # Find nearest detected enemy
        nearest_enemy = None
        nearest_range = float('inf')

        for enemy in enemies:
            if enemy.id in aircraft.detected_targets and enemy.alive:
                range_m = aircraft.position.distance_to(enemy.position)
                if range_m < nearest_range:
                    nearest_range = range_m
                    nearest_enemy = enemy

        if nearest_enemy is not None:
            # Turn toward enemy
            to_enemy = (nearest_enemy.position - aircraft.position).normalized()
            aircraft.update(self.dt, commanded_heading=to_enemy,
                          commanded_speed=aircraft.cruise_speed)

            # Consider launching missile
            if self.make_launch_decision(aircraft, nearest_enemy):
                self.launch_missile(aircraft, nearest_enemy)
        else:
            # No enemies detected, maintain course
            aircraft.update(self.dt, commanded_speed=aircraft.cruise_speed)

    def step(self):
        """Execute one simulation timestep"""
        if not self.engagement_active:
            return

        # Update detections
        self.update_detections()

        # AI control for all aircraft
        for aircraft in self.blue_force:
            self.simple_ai_control(aircraft, self.red_force)

        for aircraft in self.red_force:
            self.simple_ai_control(aircraft, self.blue_force)

        # Update missiles
        self.update_missiles()

        # Advance time
        self.time += self.dt

        # Check termination conditions
        blue_alive = sum(1 for a in self.blue_force if a.alive)
        red_alive = sum(1 for a in self.red_force if a.alive)

        if blue_alive == 0 or red_alive == 0 or self.time >= self.max_time:
            self.engagement_active = False

    def run(self, verbose: bool = True) -> Dict:
        """
        Run engagement to completion

        Args:
            verbose: Print events as they occur

        Returns:
            Dictionary with engagement results
        """
        print("=" * 80)
        print("BVR ENGAGEMENT START")
        print("=" * 80)
        print(f"Blue Force: {len(self.blue_force)} aircraft")
        print(f"Red Force: {len(self.red_force)} aircraft")
        print()

        while self.engagement_active:
            self.step()

            # Print events from this timestep
            if verbose:
                for event in self.events:
                    if event['time'] == self.time:
                        self._print_event(event)

        # Print final results
        print()
        print("=" * 80)
        print("ENGAGEMENT COMPLETE")
        print("=" * 80)

        blue_alive = sum(1 for a in self.blue_force if a.alive)
        red_alive = sum(1 for a in self.red_force if a.alive)

        print(f"Duration: {self.time:.1f} seconds")
        print()
        print(f"Blue Force: {blue_alive}/{len(self.blue_force)} survived")
        print(f"Red Force: {red_alive}/{len(self.red_force)} survived")
        print()
        print(f"Blue Kills: {self.blue_kills}")
        print(f"Red Kills: {self.red_kills}")

        if blue_alive > red_alive:
            print("\nWINNER: BLUE FORCE")
        elif red_alive > blue_alive:
            print("\nWINNER: RED FORCE")
        else:
            print("\nRESULT: DRAW")

        return {
            'duration': self.time,
            'blue_survivors': blue_alive,
            'red_survivors': red_alive,
            'blue_kills': self.blue_kills,
            'red_kills': self.red_kills,
            'events': self.events
        }

    def _print_event(self, event: Dict):
        """Print a formatted event"""
        t = event['time']
        etype = event['type']

        if etype == 'DETECTION':
            print(f"[{t:6.1f}s] {event['detector']} DETECTED {event['target']} "
                  f"at {event['range_km']:.1f} km")

        elif etype == 'MISSILE_LAUNCH':
            print(f"[{t:6.1f}s] {event['shooter']} LAUNCHED {event['missile']} "
                  f"at {event['target']} (range: {event['range_km']:.1f} km)")

        elif etype == 'KILL':
            print(f"[{t:6.1f}s] *** {event['victim']} DESTROYED "
                  f"(Pk: {event['pk']:.2f}) ***")
