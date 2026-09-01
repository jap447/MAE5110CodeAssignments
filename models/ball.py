import numpy as np

def dynamics(t, state, params):
    gravity = params["gravity"]
    height = state[0]
    velocity = state[1]
    acceleration = -gravity
    state_derivative = np.array([velocity, acceleration])
    return state_derivative

def generate_params():
    params = {
        "gravity": 9.81,  # gravity m/s^2)
        "mass": 1,  # point mass at end of rod (kg)
        "restitution_coeff": 0.8,  # damping coefficient (kg*m^2/s)
    }
    return params


def calculate_energy(state, params):
    """Compute energies for a state ``(2,)`` or trajectory ``(2, N)``."""
    gravity = params["gravity"]
    mass = params["mass"]

    position = state[0]  # indexes entire row "vectorized" if state is (2, N)
    velocity = state[1]

    kinetic_energy = 0.5 * mass * velocity ** 2
    potential_energy = mass * gravity * position
    return kinetic_energy, potential_energy
