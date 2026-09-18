import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from integrators import rk4 as integrator
from models import rimless_wheel as model

# Basic simulation of the rimless wheel

params = {
    "gravity": 9.81,  # gravity m/s^2)
    "length": 1,  # spoke length (m)
    "mass": 0.2,  # Wheel mass (kg)
    "restitution_coeff": 0.0,  # restitution coefficient (-)
    "N": 8,  # number of spokes (-)
    "Incline": np.pi / 24,  # Incline angle (rad)
}

# Some setup.
# Assume t = 0 is right after impact of a spoke, so theta = -(alpha - gamma).
N = params["N"]
gamma = params["Incline"]
alpha = np.pi / N
timestep = 1e-2
impact_timestep = 1e-5
sim_time = 20.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep

# Analysis setup
theta_values = np.linspace(-(alpha - gamma), alpha + gamma, 50)
omega_values = np.linspace(0.0, 2.5, 50)
theta_grid, omega_grid = np.meshgrid(theta_values, omega_values)
roa = model.compute_roa(
    theta_grid,
    omega_grid,
    n_timesteps,
    time_traj,
    timestep,
    impact_timestep,
    params,
    integrator,
    model,
)

cmap = ListedColormap(
    [
        "orange",  # 0 = fails before first step
        "royalblue",  # 1 = one step then fails
        "red",  # 2 = multiple steps then fails
        "purple",  # 3 = converged walking
        "gray",  # 4 = unresolved
    ]
)

norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

plt.figure(figsize=(10, 7))
image = plt.pcolormesh(
    theta_grid,
    omega_grid,
    roa,
    cmap=cmap,
    norm=norm,
    shading="auto",
)

colorbar = plt.colorbar(image, ticks=[0, 1, 2, 3, 4])
colorbar.ax.set_yticklabels(
    [
        "failed before first impact",
        "one step then failed",
        "multiple steps then failed",
        "walking limit cycle",
        "unresolved",
    ]
)

plt.xlabel(r"$\theta$ (rad)")
plt.ylabel(r"$\dot{\theta}$ (rad/s)")
plt.title("Rimless Wheel Long-Term Behavior")
plt.tight_layout()
plt.show()
