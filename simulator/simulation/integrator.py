"""
Numerical integration for flight dynamics
Uses 4th-order Runge-Kutta (RK4) for accuracy and stability
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable
from ..physics.vector import Vector3
from ..physics.quaternion import Quaternion


@dataclass
class FlightState:
    """
    Complete flight state for 6-DOF simulation

    Position and velocity in world (NED) frame
    Orientation as quaternion
    Angular velocity in body frame
    """
    # Translational state (world/NED frame)
    position: Vector3          # Position (m)
    velocity: Vector3          # Velocity (m/s)

    # Rotational state
    orientation: Quaternion    # Orientation quaternion (world to body)
    angular_velocity: Vector3  # Angular velocity in body frame (rad/s)

    # Time
    time: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert state to numpy array for integration"""
        return np.array([
            # Position (3)
            self.position.x, self.position.y, self.position.z,
            # Velocity (3)
            self.velocity.x, self.velocity.y, self.velocity.z,
            # Orientation quaternion (4)
            self.orientation.w, self.orientation.x,
            self.orientation.y, self.orientation.z,
            # Angular velocity (3)
            self.angular_velocity.x, self.angular_velocity.y,
            self.angular_velocity.z
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray, time: float = 0.0) -> 'FlightState':
        """Create state from numpy array"""
        return cls(
            position=Vector3(arr[0], arr[1], arr[2]),
            velocity=Vector3(arr[3], arr[4], arr[5]),
            orientation=Quaternion(arr[6], arr[7], arr[8], arr[9]).normalized(),
            angular_velocity=Vector3(arr[10], arr[11], arr[12]),
            time=time
        )

    def clone(self) -> 'FlightState':
        """Create a deep copy"""
        return FlightState(
            position=self.position.clone(),
            velocity=self.velocity.clone(),
            orientation=self.orientation.clone(),
            angular_velocity=self.angular_velocity.clone(),
            time=self.time
        )


class StateDerivative:
    """Time derivatives of flight state"""

    def __init__(self):
        self.position_dot = Vector3.zero()      # Velocity
        self.velocity_dot = Vector3.zero()      # Acceleration
        self.orientation_dot = Quaternion.identity()  # Quaternion derivative
        self.angular_velocity_dot = Vector3.zero()    # Angular acceleration

    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([
            self.position_dot.x, self.position_dot.y, self.position_dot.z,
            self.velocity_dot.x, self.velocity_dot.y, self.velocity_dot.z,
            self.orientation_dot.w, self.orientation_dot.x,
            self.orientation_dot.y, self.orientation_dot.z,
            self.angular_velocity_dot.x, self.angular_velocity_dot.y,
            self.angular_velocity_dot.z
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'StateDerivative':
        """Create from numpy array"""
        deriv = cls()
        deriv.position_dot = Vector3(arr[0], arr[1], arr[2])
        deriv.velocity_dot = Vector3(arr[3], arr[4], arr[5])
        deriv.orientation_dot = Quaternion(arr[6], arr[7], arr[8], arr[9])
        deriv.angular_velocity_dot = Vector3(arr[10], arr[11], arr[12])
        return deriv


class RK4Integrator:
    """4th-order Runge-Kutta integrator for flight dynamics"""

    @staticmethod
    def compute_state_derivative(state: FlightState, force: Vector3, moment: Vector3,
                                 mass: float, inertia: np.ndarray) -> StateDerivative:
        """
        Compute time derivative of state given forces and moments

        Args:
            state: Current flight state
            force: Total force in world frame (N)
            moment: Total moment in body frame (N·m)
            mass: Aircraft mass (kg)
            inertia: Moment of inertia tensor (kg·m²) in body frame

        Returns:
            State derivative
        """
        deriv = StateDerivative()

        # Position derivative = velocity
        deriv.position_dot = state.velocity.clone()

        # Velocity derivative = acceleration (F/m in world frame)
        deriv.velocity_dot = force / mass

        # Orientation derivative from angular velocity
        # q_dot = 0.5 * q * omega (where omega is quaternion [0, wx, wy, wz])
        omega_quat = Quaternion(0,
                               state.angular_velocity.x,
                               state.angular_velocity.y,
                               state.angular_velocity.z)
        deriv.orientation_dot = state.orientation * omega_quat
        deriv.orientation_dot.data *= 0.5

        # Angular acceleration from Euler's rotation equations
        # I * ω_dot + ω × (I * ω) = M
        omega = state.angular_velocity.to_array()
        I_omega = inertia @ omega
        omega_cross_I_omega = np.cross(omega, I_omega)

        # ω_dot = I^(-1) * (M - ω × (I * ω))
        I_inv = np.linalg.inv(inertia)
        alpha = I_inv @ (moment.to_array() - omega_cross_I_omega)

        deriv.angular_velocity_dot = Vector3(alpha[0], alpha[1], alpha[2])

        return deriv

    @staticmethod
    def step(state: FlightState, dt: float,
            derivative_func: Callable[[FlightState, float], StateDerivative]) -> FlightState:
        """
        Perform one RK4 integration step

        Args:
            state: Current state
            dt: Time step (s)
            derivative_func: Function that computes state derivative
                            Takes (state, time) and returns StateDerivative

        Returns:
            New state after time step
        """
        # Get current state as array
        y = state.to_array()
        t = state.time

        # RK4 stages
        k1 = derivative_func(state, t).to_array()

        state2 = FlightState.from_array(y + 0.5 * dt * k1, t + 0.5 * dt)
        k2 = derivative_func(state2, t + 0.5 * dt).to_array()

        state3 = FlightState.from_array(y + 0.5 * dt * k2, t + 0.5 * dt)
        k3 = derivative_func(state3, t + 0.5 * dt).to_array()

        state4 = FlightState.from_array(y + dt * k3, t + dt)
        k4 = derivative_func(state4, t + dt).to_array()

        # Combine stages
        y_new = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Create new state
        new_state = FlightState.from_array(y_new, t + dt)

        # Re-normalize quaternion to prevent drift
        new_state.orientation = new_state.orientation.normalized()

        return new_state

    @staticmethod
    def integrate(state: FlightState, t_end: float, dt: float,
                 derivative_func: Callable[[FlightState, float], StateDerivative],
                 callback: Callable[[FlightState], bool] = None) -> list[FlightState]:
        """
        Integrate from current state to t_end

        Args:
            state: Initial state
            t_end: End time (s)
            dt: Time step (s)
            derivative_func: Function computing state derivative
            callback: Optional callback function called after each step.
                     Returns True to continue, False to stop.

        Returns:
            List of states at each time step
        """
        states = [state.clone()]
        current_state = state.clone()

        while current_state.time < t_end:
            # Adjust last step if needed
            step_dt = min(dt, t_end - current_state.time)

            # Take step
            current_state = RK4Integrator.step(current_state, step_dt, derivative_func)
            states.append(current_state.clone())

            # Call callback if provided
            if callback is not None:
                if not callback(current_state):
                    break

        return states
