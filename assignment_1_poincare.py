import matplotlib.pyplot as plt
import numpy as np

from integrators import rk4 as integrator
from models import rimless_wheel as model

# Parameters for the rimless wheel
params = {
    "gravity": 9.81,  # gravity m/s^2)
    "length": 1,  # spoke length (m)
    "mass": 0.2,  # Wheel mass (kg)
    "restitution_coeff": 0.0,  # restitution coefficient (-)
    "N": 8,  # number of spokes (-)
    "Incline": np.pi/24,  # Incline angle (rad)
}

N = params["N"]
gamma = params["Incline"]
alpha = np.pi / N
post_impact_angle = alpha - gamma
step_drop = 2 * params["length"] * np.sin(alpha) * np.sin(gamma)

# Initial condition: right after impact, so theta = -(alpha - gamma)
initial_state = np.array([-(post_impact_angle), 1.0], dtype=float)

# Time-stepping parameters: use a smaller dt for contact detection accuracy.
timestep = 1e-3
sim_time = 10.0
impact_timestep = 1e-6

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state

height_traj = np.zeros(n_timesteps)
impact_velocities = []
step_count = 0

# Simulate the wheel and collect the post-impact angular velocities.
for step, t in enumerate(time_traj[:-1]):
    current_state = state_traj[:, step]
    next_state = integrator(model.pendulum_dynamics, t, current_state, timestep, params)
    state_traj[:, step + 1] = next_state
    height_traj[step + 1] = height_traj[step]

    if model.detect_event(current_state, next_state, params):
        refined_state = model.refine_impact(
            current_state,
            t,
            timestep,
            impact_timestep,
            params,
            integrator,
        )
        state_traj[:, step + 1] = model.reset_impact(refined_state, params)
        impact_velocities.append(float(state_traj[1, step + 1]))

        step_count += 1
        height_traj[step + 1] = -step_count * step_drop


# Energy Sanity check

kinetic_energy, potential_energy = model.calculate_energy(state_traj, height_traj, params)
plt.figure()
plt.plot(time_traj, potential_energy, label="Potential energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
plt.plot(time_traj, kinetic_energy + potential_energy, label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Rimless wheel energy")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.semilogx(epsilons, floquet_estimates, "o-")
plt.xlabel(r"$\epsilon$")
plt.ylabel("Floquet multiplier")
plt.grid(True)
plt.show()