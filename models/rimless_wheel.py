import numpy as np


def pendulum_dynamics(t, state, params):
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    angle = state[0]
    angular_velocity = state[1]

    angular_acceleration = (
        mass * gravity * length * np.sin(angle)
    ) / (mass * length**2)

    state_derivative = np.array([angular_velocity, angular_acceleration])
    return state_derivative


def detect_event(state, params):
    touch_angle = np.pi / params["N"] + params["Incline"]
    return state[0] >= touch_angle


def reset_impact(state, params):
    alpha = np.pi / params["N"]
    touch_angle = alpha - params["Incline"]
    reset_state = np.array(state, copy=True)
    reset_state[0] = -touch_angle
    reset_state[1] = np.cos(2 * alpha) * state[1]
    return reset_state


def calculate_energy(state, height_traj, params):
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    angle = state[0]  # indexes entire row "vectorized" if state is (2, N)
    angular_velocity = state[1]
    
    kinetic_energy = 0.5 * mass * (length * angular_velocity) ** 2
    potential_energy = mass * gravity * (length * np.cos(angle) + height_traj)
    return kinetic_energy, potential_energy
