import numpy as np
import matplotlib.pyplot as plt

from models import rimless_wheel as model
from integrators import rk4 as integrator

# Basic simulation of the rimless wheel

params = {
    "gravity": 9.81,  # gravity m/s^2)
    "length": 1,  # spoke length (m)
    "mass": 0.2,  # Wheel mass (kg)
    "restitution_coeff": 0.0,  # restitution coefficient (-)
    "N": 8,  # number of spokes (-)
    "Incline": np.pi/6,  # Incline angle (rad)
}

# some set-up
# Assume t=0 right after the impact of a spoke. Thus, theta = -(alpha - gamma)

N = params["N"]
gamma = params["Incline"]
alpha = np.pi / N

initial_state = np.array([-(alpha - gamma), 1.0])  # [theta, theta_dot]

timestep = 1e-5
sim_time = 10.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state

# simulation
# First, the system behaves like a simple pendulum until impact of the next spoke.
for step, t in enumerate(time_traj[:-1]):
    state_traj[:, step + 1] = state_traj[:, step] + timestep * model.dynamics(
        t, state_traj[:, step], params
    )


# Energy Sanity check: since there is no actuation, and no damping, total energy should stay constant. If we turn on the damping coefficient, it should slowly bleed out energy until it comes to a stand-still.

kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)
total_energy = kinetic_energy + potential_energy