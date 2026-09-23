from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from integrators import rk4 as integrator
from models import inverted_pendulum_walker as model

# Fixed controls for this visualization example.
params = {
    "gravity": 9.81,  # m/s^2
    "length": 1.0,  # m
    "mass": 1.0,  # kg
    "incline": 0.06,  # rad
    "angle_of_attack": np.pi / 8,  # rad
    "ankle_torque": 0.0,  # N m
    "torque_min": -10.0,  # N m
    "torque_max": 10.0,  # N m
    "Kp": 100.0,  # s^-2
    "Kd": 10.0,  # s^-1
}

mgl = params["mass"] * params["gravity"] * params["length"]
params["torque_min"] = -0.1 * mgl
params["torque_max"] = 0.05 * mgl

initial_state = np.array([0.0, 3.0])
timestep = 1e-4
sim_time = 3.0

# Simulate with RK4 and refine integration near impact.
def simulate(
    initial_state,
    params,
    timestep,
    sim_time,
    balance=False,
    *,
    impact_timestep=1e-5,
    desired_number_of_steps=None,
    roa_data=None,
):

    params = params.copy()
    time_traj = np.append(np.arange(0.0, sim_time, timestep), sim_time)
    state_traj = np.zeros((2, len(time_traj)))
    state_traj[:, 0] = initial_state
    completed_steps = 0

    for step, t in enumerate(time_traj[:-1]):
        current_state = state_traj[:, step]
        if not balance and roa_data is not None:
            if reached_roa(current_state, *roa_data):
                balance = True
        dt = time_traj[step + 1] - t
        params["ankle_torque"] = (
            compute_ankle_torque(current_state, params) if balance else 0.0
        )

        next_state = integrator(model.dynamics, t, current_state, dt, params)
        state_traj[:, step + 1] = next_state

        if balance == False and model.event_guard(current_state, next_state, params):
            impact_state, elapsed, contact = model.refine_impact(
                current_state, t, dt, impact_timestep, params, integrator
            )
            state_traj[:, step + 1] = impact_state

            if not contact:
                continue

            state_traj[:, step + 1] = model.event_dynamics(impact_state, params)
            completed_steps += 1

            if roa_data is not None:
                if reached_roa(state_traj[:, step + 1], *roa_data):
                    balance = True
                    params["ankle_torque"] = compute_ankle_torque(
                        state_traj[:, step + 1], params
                    )

            if completed_steps == desired_number_of_steps:
                time_traj[step + 1] = t + elapsed
                return (
                    time_traj[:step + 2],
                    state_traj[:, :step + 2],
                    completed_steps,
                )

            remaining_time = dt - elapsed
            if remaining_time > 0:
                state_traj[:, step + 1] = integrator(
                    model.dynamics,
                    t + elapsed,
                    state_traj[:, step + 1],
                    remaining_time,
                    params,
                )

    return time_traj, state_traj, completed_steps

fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

def compute_ankle_torque(state, params):
    theta, angular_velocity = state
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]
    torque_min =  params["torque_min"]
    torque_max =  params["torque_max"]
    Kp = params["Kp"]
    Kd = params["Kd"]

    gravity_torque = -mass * gravity * length * np.sin(theta)
    stabilization = -mass * length**2 * (Kp * theta + Kd * angular_velocity)
    ankle_torque = gravity_torque + stabilization
    return np.clip(ankle_torque, torque_min, torque_max)


def classify_balance(
    time_traj,
    state_traj,
    params,
    theta_tolerance=1e-3,
    omega_tolerance=1e-3,
    settling_window=1.0,
):
    """Return 0 for failure, 1 for convergence, or 2 for unresolved."""
    theta, omega = state_traj
    failed = (
        not np.all(np.isfinite(state_traj))
        or np.any(np.cos(theta - params["incline"]) <= 0.0)
    )
    if failed:
        return 0
    if time_traj[-1] < settling_window:
        return 2

    final_window = time_traj >= time_traj[-1] - settling_window
    converged = (
        np.all(np.abs(theta[final_window]) < theta_tolerance)
        and np.all(np.abs(omega[final_window]) < omega_tolerance)
    )
    return 1 if converged else 2


def estimate_roa(theta_values, omega_values, params, timestep, sim_time):
    """Classify balancing trajectories over a grid of initial states."""
    roa = np.zeros((len(omega_values), len(theta_values)), dtype=int)
    for i, omega in enumerate(omega_values):
        for j, theta in enumerate(theta_values):
            time_traj, state_traj, _ = simulate(
                [theta, omega], params, timestep, sim_time, balance=True,
            )
            roa[i, j] = classify_balance(time_traj, state_traj, params)
    return roa


def reached_roa(state, theta_values, omega_values, roa):
    """Accept only cells whose four corners all converged."""
    theta, omega = state
    if not np.all(np.isfinite(state)):
        return False
    inside_grid = (
        theta_values[0] <= theta <= theta_values[-1]
        and omega_values[0] <= omega <= omega_values[-1]
    )
    if not inside_grid:
        return False

    j = np.searchsorted(theta_values, theta, side="right") - 1
    i = np.searchsorted(omega_values, omega, side="right") - 1
    j = min(j, len(theta_values) - 2)
    i = min(i, len(omega_values) - 2)
    return bool(np.all(roa[i:i + 2, j:j + 2] == 1))


def plot_roa(theta_values, omega_values, roa):
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap(["tomato", "seagreen", "lightgray"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(layout="constrained")
    image = ax.pcolormesh(
        theta_values, omega_values, roa,
        cmap=cmap, norm=norm, shading="nearest",
    )
    colorbar = fig.colorbar(image, ax=ax, ticks=[0, 1, 2])
    colorbar.ax.set_yticklabels(["Failed", "Converged", "Unresolved"])
    ax.set(
        xlabel=r"$\theta_0$ (rad)",
        ylabel=r"$\dot{\theta}_0$ (rad/s)",
        title="Estimated balancing region of attraction",
    )
    return fig, ax


def draw_frame(index):
    # The massless swing leg is repositioned instantaneously at each impact.
    model.visualize(state_traj[:, index], params, ax=ax)
    ax.set_title(f"t = {time_traj[index]:.2f} s")


# Wider domain with 51 * 49 = 2,499 initial conditions, including upright rest.
theta_values = np.linspace(-0.4, 0.4, 50)
omega_values = np.linspace(-1.5, 1.5, 50)
roa = estimate_roa(
    theta_values, omega_values, params, timestep=0.005, sim_time=5.0,
)
_, roa_ax = plot_roa(theta_values, omega_values, roa)
roa_ax.set_title("Estimated balancing RoA: initial sweep")

# Mark both sides of each classification boundary and all unresolved points.
boundary = np.zeros(roa.shape, dtype=bool)
row_changes = roa[1:, :] != roa[:-1, :]
boundary[1:, :] |= row_changes
boundary[:-1, :] |= row_changes
column_changes = roa[:, 1:] != roa[:, :-1]
boundary[:, 1:] |= column_changes
boundary[:, :-1] |= column_changes

check_points = boundary | (roa == 2)
checked_roa = roa.copy()
for i, j in np.argwhere(check_points):
    time_check, state_check, _ = simulate(
        [theta_values[j], omega_values[i]],
        params,
        timestep=0.0025,
        sim_time=10.0,
        balance=True,
    )
    checked_roa[i, j] = classify_balance(time_check, state_check, params)

changed = np.count_nonzero(checked_roa != roa)
print(f"Rechecked {np.count_nonzero(check_points)} grid points.")
print(f"Changed classifications: {changed}")
roa = checked_roa
_, checked_ax = plot_roa(theta_values, omega_values, roa)
checked_ax.set_title("Estimated balancing RoA: boundary points rechecked")

# Check the final map for a successful region cut off by the grid limits.
converged = roa == 1
touches_boundary = (
    np.any(converged[0, :])
    or np.any(converged[-1, :])
    or np.any(converged[:, 0])
    or np.any(converged[:, -1])
)
print(f"Converged region reaches grid boundary: {touches_boundary}")

balance_time, balance_state, _ = simulate(
    [0.01, 0.0], params, timestep=0.001, sim_time=5.0, balance=True,
)
balance_fig, axes = plt.subplots(2, 1, sharex=True, layout="constrained")
axes[0].plot(balance_time, balance_state[0])
axes[0].set_ylabel(r"$\theta$ (rad)")
axes[1].plot(balance_time, balance_state[1])
axes[1].set_ylabel(r"$\dot{\theta}$ (rad/s)")
axes[1].set_xlabel("Time (s)")

time_traj, state_traj, completed_steps = simulate(
    initial_state, params, timestep, sim_time,
    roa_data=(theta_values, omega_values, roa),
)


# Simulate at a small timestep, but render only 25 frames per second.
fps = 25
frame_stride = round(1 / (fps * timestep))
frame_indices = list(range(0, time_traj.size, frame_stride))
if frame_indices[-1] != time_traj.size - 1:
    frame_indices.append(time_traj.size - 1)

animation = FuncAnimation(
    fig, draw_frame, frames=frame_indices, interval=1000 / fps, repeat=False
)
output = Path("output/assignment_2")
output.mkdir(parents=True, exist_ok=True)
animation.save(output / "walker.gif", writer=PillowWriter(fps=fps))

# To save an MP4 instead, install FFmpeg and use:
# animation.save(output / "walker.mp4", writer="ffmpeg", fps=fps)
print(f"Saved {output / 'walker.gif'} ({completed_steps} footstrikes).")
plt.show()
