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
    "Incline": np.pi / 24,  # Incline angle (rad)
}

# some set-up
# Assume t=0 right after the impact of a spoke. Thus, theta = -(alpha - gamma)

N = params["N"]
gamma = params["Incline"]
alpha = np.pi / N
touch_angle = alpha - gamma
step_drop = 2 * params["length"] * np.sin(alpha) * np.sin(gamma)
initial_state = np.array([-(touch_angle), 1])  # [theta, theta_dot]

timestep = 1e-4
sim_time = 5.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state

height_traj = np.zeros(n_timesteps)
step_count = 0

# simulation
# First, the system behaves like a simple pendulum until impact of the next spoke.

for step, t in enumerate(time_traj[:-1]):
    state_traj[:, step + 1] = integrator(
        model.pendulum_dynamics,
        t,
        state_traj[:, step],
        timestep,
        params,
    )
    height_traj[step + 1] = height_traj[step]

    # Handle bounces after the step completes
    if model.detect_event(state_traj[:, step], state_traj[:, step + 1], params):
        state_traj[:, step + 1] = model.reset_impact(state_traj[:, step + 1], params)

        step_count += 1
        height_traj[step + 1] = -step_count * step_drop


# Energy Sanity check

kinetic_energy, potential_energy = model.calculate_energy(
    state_traj, height_traj, params
)
total_energy = kinetic_energy + potential_energy

plt.figure()
plt.plot(time_traj, potential_energy, label="Potential energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Rimless wheel energy")
plt.legend()
plt.tight_layout()
plt.show()
