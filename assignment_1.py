import matplotlib.pyplot as plt
import numpy as np

from integrators import rk4 as integrator
from models import rimless_wheel as model

# Basic simulation of the rimless wheel

params = {
    "gravity": 9.81,  # gravity m/s^2)
    "length": 1,  # spoke length (m)
    "mass": 0.2,  # Wheel mass (kg)
    "restitution_coeff": 0.0,  # restitution coefficient (-)
    "N": 8,  # number of spokes (-)
    "Incline": np.pi/24,  # Incline angle (rad)
}

# some set-up
# Assume t=0 right after the impact of a spoke. Thus, theta = -(alpha - gamma)

N = params["N"]
gamma = params["Incline"]
alpha = np.pi / N
post_impact_angle = alpha - gamma
step_drop = 2 * params["length"] * np.sin(alpha) * np.sin(gamma)
initial_state = np.array([-(post_impact_angle), 1])  # [theta, theta_dot]

timestep = 1e-2
impact_timestep = 1e-5
sim_time = 15.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep

# Analysis setup
# theta_values = np.linspace(-1, 1, 50)
theta_values = np.linspace(-(alpha - gamma), alpha + gamma,20)
omega_values = np.linspace(0.0, 5.0, 20)
theta_grid, omega_grid = np.meshgrid(theta_values, omega_values)
roa = np.zeros_like(theta_grid, dtype=int)  # region of attraction
# 0 = failed/no walking
# 1 = converged walking limit cycle
# 2 = transient/undecided

# simulation
# First, the system behaves like a simple pendulum until impact of the next spoke.

for i in range(theta_grid.shape[0]):
    for j in range(theta_grid.shape[1]):

        state_traj = np.zeros((2, n_timesteps))
        height_traj = np.zeros(n_timesteps)

        state_traj[:, 0] = [theta_grid[i, j], omega_grid[i, j]]

        impact_velocities = []
        converged = False

        for step, t in enumerate(time_traj[:-1]):

            next_state = integrator(model.pendulum_dynamics, t, state_traj[:, step],
                timestep, params)

            state_traj[:, step + 1] = next_state

            if model.detect_event(state_traj[:, step], next_state, params):

                impact_state = model.refine_impact(state_traj[:, step], t, timestep,
                    impact_timestep, params, integrator)

                state_traj[:, step + 1] = model.reset_impact(impact_state, params)

                impact_velocities.append(state_traj[1, step + 1])

                if model.walking_converged(impact_velocities):
                    converged = True
                    break

        if converged:
            roa[i, j] = 1
        else:
            roa[i, j] = 0

# Energy Sanity check

kinetic_energy, potential_energy = model.calculate_energy(state_traj, height_traj, params)
total_energy = kinetic_energy + potential_energy

from matplotlib.colors import BoundaryNorm, ListedColormap

plt.figure()

plt.pcolormesh(theta_grid, omega_grid, roa, shading="auto")
plt.xlabel(r"$\theta$ (rad)")
plt.ylabel(r"$\dot{\theta}$ (rad/s)")
plt.title("Estimated Region of Attraction")

plt.show()