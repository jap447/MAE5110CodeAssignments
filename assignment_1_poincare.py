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
    "Incline": np.pi / 24,  # Incline angle (rad)
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

# Build the one-dimensional Poincaré return map.
impact_velocities = np.asarray(impact_velocities, dtype=float)
if impact_velocities.size < 2:
    raise ValueError("Not enough impacts were detected to construct the Poincaré map.")

v_n = impact_velocities[:-1]
v_np1 = impact_velocities[1:]

# Estimate the fixed point
fixed_point_index = np.argmin(np.abs(v_n - v_np1))
fixed_point = 0.5 * (v_n[fixed_point_index] + v_np1[fixed_point_index])

epsilons = [5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]

floquet_estimates = []

for epsilon in epsilons:
    floquet = model.estimate_floquet(
        fixed_point,
        epsilon,
        params,
        timestep,
        impact_timestep,
        integrator,
        post_impact_angle,
    )
    floquet_estimates.append(floquet)

    print(f"Epsilon: {epsilon:.5f}, Floquet estimate: {floquet:.6f}")

# Estimate the local slope near the fixed point:
# mu ≈ (P(v + eps) - P(v - eps)) / (2 eps)
# This is the approximate Floquet multiplier for the rolling limit cycle.
eps = 1e-3
mu = model.estimate_floquet(
    fixed_point, eps, params, timestep, impact_timestep, integrator, post_impact_angle
)


print(f"Estimated fixed point: {fixed_point:.6f} rad/s")
print(f"Estimated Floquet multiplier: {mu:.6f}")

# Plot the return map and identity line.
plt.figure(figsize=(8, 8))
plt.scatter(v_n, v_np1, s=12, alpha=0.7, label="Poincaré map samples")
min_val = min(np.min(v_n), np.min(v_np1))
max_val = max(np.max(v_n), np.max(v_np1))
identity = np.linspace(min_val, max_val, 200)
plt.plot(identity, identity, "k--", label="identity")
plt.axvline(
    fixed_point,
    color="tab:red",
    linestyle="-.",
    linewidth=1.5,
    label=f"fixed point = {fixed_point:.4f}",
)
plt.axhline(fixed_point, color="tab:red", linestyle="-.", linewidth=1.5)
plt.xlabel(r"$\dot{\theta}_n$ (rad/s)")
plt.ylabel(r"$\dot{\theta}_{n+1}$ (rad/s)")
plt.title("Rimless Wheel Poincaré Return Map")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Energy sanity check.
kinetic_energy, potential_energy = model.calculate_energy(
    state_traj, height_traj, params
)
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
