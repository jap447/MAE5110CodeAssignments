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
sim_time = 20.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep

# Analysis setup
# theta_values = np.linspace(-1, 1, 50)
theta_values = np.linspace(-(alpha - gamma), alpha + gamma,50)
omega_values = np.linspace(0.0, 2.5, 50)
theta_grid, omega_grid = np.meshgrid(theta_values, omega_values)
roa = np.zeros_like(theta_grid, dtype=int)  # region of attraction
# 0 = failed before first impact
# 1 = took exactly one step, then failed
# 2 = took multiple steps, then failed
# 3 = converged to period-1 walking
# 4 = other / unresolved

# simulation
# First, the system behaves like a simple pendulum until impact of the next spoke.

for i in range(theta_grid.shape[0]):
    for j in range(theta_grid.shape[1]):

        state_traj = np.zeros((2, n_timesteps))
        height_traj = np.zeros(n_timesteps)

        state_traj[:, 0] = [theta_grid[i, j], omega_grid[i, j]]

        impact_velocities = []
        converged = False
        failed  = False

        for step, t in enumerate(time_traj[:-1]):

            current_state = state_traj[:, step]
            
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
            else:
                omega_current = current_state[1]
                omega_next = next_state[1]

                if (omega_current >= 0 and omega_next < 0.0):
                    failed = True
                    break
        if converged:
            clasification = 3
        elif failed:
            n_impacts  = len(impact_velocities)
            if n_impacts == 0:
                clasification = 0
            elif n_impacts == 1:
                clasification = 1
            else:
                clasification = 2
        else:
            clasification = 4

        roa[i, j] = clasification


# Energy Sanity check

kinetic_energy, potential_energy = model.calculate_energy(state_traj, height_traj, params)
total_energy = kinetic_energy + potential_energy

from matplotlib.colors import BoundaryNorm, ListedColormap

cmap = ListedColormap([
    "orange",       # 0 - fails before first step
    "royalblue",   # 1 - one step then fails
    "red",      # 2 - multiple steps then fails
    "seagreen",    # 3 - converged walking
    "gray"         # 4 - unresolved
])

norm = BoundaryNorm(
    [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
    cmap.N
)

plt.figure(figsize=(10, 7))

image = plt.pcolormesh(
    theta_grid,
    omega_grid,
    roa,
    cmap=cmap,
    norm=norm,
    shading="auto"
)

colorbar = plt.colorbar(
    image,
    ticks=[0, 1, 2, 3, 4]
)

colorbar.ax.set_yticklabels([
    "failed before first impact",
    "one step then failed",
    "multiple steps then failed",
    "walking limit cycle",
    "unresolved"
])

plt.xlabel(r"$\theta$ (rad)")
plt.ylabel(r"$\dot{\theta}$ (rad/s)")
plt.title("Rimless Wheel Long-Term Behavior")

plt.tight_layout()
plt.show()