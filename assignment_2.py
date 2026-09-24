import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from assignment_2_validation import compare_policy_grids
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

        if not balance and model.event_guard(current_state, next_state, params):
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


def evaluate_transition(
    initial_velocity,
    angle_of_attack,
    params,
    roa_data,
    timestep=0.001,
    max_time=10.0,
    impact_timestep=1e-5,
):

    params = params.copy()
    params["angle_of_attack"] = angle_of_attack
    params["ankle_torque"] = 0.0
    state = np.array([0.0, initial_velocity], dtype=float)
    t = 0.0
    footstrikes = 0

    while t < max_time:
        if reached_roa(state, *roa_data):
            return "success", np.nan, footstrikes
        if (
            not np.all(np.isfinite(state))
            or np.cos(state[0] - params["incline"]) <= 0.0
            or state[1] <= 0.0
        ):
            return "failure", np.nan, footstrikes

        dt = min(timestep, max_time - t)
        next_state = integrator(model.dynamics, t, state, dt, params)
        crossed_section = (
            state[0] < 0.0 <= next_state[0] and next_state[1] > 0.0
        )
        if crossed_section:
            # Estimate theta = 0 between the two RK4 samples.
            fraction = -state[0] / (next_state[0] - state[0])
            section_velocity = state[1] + fraction * (next_state[1] - state[1])
            section_state = np.array([0.0, section_velocity])
            if reached_roa(section_state, *roa_data):
                return "success", np.nan, footstrikes
            return "return", float(section_state[1]), footstrikes

        if model.event_guard(state, next_state, params):
            impact_state, elapsed, contact = model.refine_impact(
                state, t, dt, impact_timestep, params, integrator
            )
            if reached_roa(impact_state, *roa_data):
                return "success", np.nan, footstrikes
            if contact:
                state = model.event_dynamics(impact_state, params)
                footstrikes += 1
            else:
                state = impact_state
            t += elapsed
        else:
            state = next_state
            t += dt

    if reached_roa(state, *roa_data):
        return "success", np.nan, footstrikes
    return "unresolved", np.nan, footstrikes


def build_transition_table(
    velocity_values,
    alpha_values,
    params,
    roa_data,
    timestep=0.001,
    max_time=10.0,
    impact_timestep=1e-5,
):
    """Build a state-action table indexed by [velocity, angle_of_attack]."""
    velocity_values = np.asarray(velocity_values, dtype=float)
    alpha_values = np.asarray(alpha_values, dtype=float)
    shape = (len(velocity_values), len(alpha_values))
    outcomes = np.full(shape, "unresolved", dtype="<U10")
    next_velocities = np.full(shape, np.nan)
    step_costs = np.zeros(shape, dtype=int)
    next_indices = np.full(shape, -1, dtype=int)

    for i, velocity in enumerate(velocity_values):
        for j, alpha in enumerate(alpha_values):
            outcome, next_velocity, footstrikes = evaluate_transition(
                velocity, alpha, params, roa_data,
                timestep=timestep,
                max_time=max_time,
                impact_timestep=impact_timestep,
            )
            outcomes[i, j] = outcome
            next_velocities[i, j] = next_velocity
            step_costs[i, j] = footstrikes
            if outcome == "return":
                if velocity_values[0] <= next_velocity <= velocity_values[-1]:
                    next_indices[i, j] = np.argmin(
                        np.abs(velocity_values - next_velocity)
                    )
                else:
                    outcomes[i, j] = "out_of_map"
        print(f"Completed velocity row {i + 1}/{len(velocity_values)}")

    return {
        "velocity_values": velocity_values,
        "alpha_values": alpha_values,
        "outcomes": outcomes,
        "next_velocities": next_velocities,
        "step_costs": step_costs,
        "next_indices": next_indices,
    }


def compute_minimum_step_policy(table):
    """Find minimum footstep costs on the discretized transition graph."""
    outcomes = table["outcomes"]
    costs = table["step_costs"]
    next_indices = table["next_indices"]
    n_states, n_actions = outcomes.shape
    steps = np.full(n_states, np.inf)
    action_costs = np.full((n_states, n_actions), np.inf)
    success = outcomes == "success"
    returns = outcomes == "return"

    # Propagate terminal success backward; shortest paths need no cycles.
    for _ in range(n_states):
        action_costs.fill(np.inf)
        action_costs[success] = costs[success]
        action_costs[returns] = costs[returns] + steps[next_indices[returns]]
        updated_steps = np.min(action_costs, axis=1)
        if np.array_equal(updated_steps, steps):
            break
        steps = updated_steps

    action_costs.fill(np.inf)
    action_costs[success] = costs[success]
    action_costs[returns] = costs[returns] + steps[next_indices[returns]]
    action_indices = np.full(n_states, -1, dtype=int)
    selected_angles = np.full(n_states, np.nan)
    for i in range(n_states):
        if not np.isfinite(steps[i]):
            continue
        best_actions = np.flatnonzero(action_costs[i] == steps[i])
        # Select a tested angle inside the widest contiguous group of ties.
        groups = np.split(
            best_actions, np.flatnonzero(np.diff(best_actions) > 1) + 1,
        )
        widest_group = max(groups, key=len)
        j = widest_group[len(widest_group) // 2]
        action_indices[i] = j
        selected_angles[i] = table["alpha_values"][j]

    return {
        "velocity_values": table["velocity_values"],
        "steps": steps,
        "action_indices": action_indices,
        "angles": selected_angles,
    }


def rollout_policy(
    initial_velocity, policy, params, roa_data,
    timestep=0.0005, max_transitions=100,
):
    """Execute a policy using actual section velocities, not rounded returns.

    Return (status, footstrikes, section_history); this is not a full time
    trajectory. A transition cap means unresolved, not physical failure.
    """
    velocities = policy["velocity_values"]
    velocity = float(initial_velocity)
    total_steps = 0
    history = []
    for _ in range(max_transitions):
        if reached_roa([0.0, velocity], *roa_data):
            return "success", total_steps, history
        if not velocities[0] <= velocity <= velocities[-1]:
            return "out_of_map", total_steps, history
        i = np.argmin(np.abs(velocities - velocity))
        if policy["action_indices"][i] < 0:
            return "no_policy", total_steps, history

        alpha = policy["angles"][i]
        outcome, next_velocity, footstrikes = evaluate_transition(
            velocity, alpha, params, roa_data, timestep=timestep,
        )
        history.append({
            "velocity": velocity,
            "alpha": alpha,
            "outcome": outcome,
            "next_velocity": next_velocity,
            "footstrikes": footstrikes,
        })
        total_steps += footstrikes
        if outcome != "return":
            return outcome, total_steps, history
        velocity = next_velocity

    return "unresolved", total_steps, history


def plot_minimum_step_policy(policy):
    velocities = policy["velocity_values"]
    reachable = np.isfinite(policy["steps"])
    needs_action = reachable & (policy["steps"] > 0)
    fig, axes = plt.subplots(
        2, 1, sharex=True, figsize=(8, 6), layout="constrained",
    )
    axes[0].plot(
        velocities, np.where(reachable, policy["steps"], np.nan),
        "o-", markersize=3,
    )
    axes[0].set_ylabel("Minimum footsteps")
    axes[0].set_title("Predicted footsteps and selected landing angle")
    axes[1].plot(
        velocities, np.where(needs_action, policy["angles"], np.nan),
        "o-", markersize=3,
    )
    axes[1].set(
        xlabel=r"Initial velocity $\dot{\theta}$ (rad/s)",
        ylabel=r"Selected $\alpha$ (rad)",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    return fig, axes


def plot_transition_table(table):
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    velocities = table["velocity_values"]
    angles = table["alpha_values"]
    outcomes = table["outcomes"]
    categories = [
        ("success", "Reached balancing RoA", "seagreen"),
        ("return", "Returned to section", "royalblue"),
        ("failure", "Forward walking failed", "tomato"),
        ("unresolved", "Time limit reached", "lightgray"),
        ("out_of_map", "Return outside velocity grid", "purple"),
    ]

    outcome_codes = np.full(outcomes.shape, -1, dtype=int)
    for code, (name, _, _) in enumerate(categories):
        outcome_codes[outcomes == name] = code
    if np.any(outcome_codes < 0):
        raise ValueError("Table contains an unknown outcome.")

    cmap = ListedColormap([color for _, _, color in categories])
    norm = BoundaryNorm(np.arange(len(categories) + 1) - 0.5, cmap.N)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    axes[0].pcolormesh(
        velocities, angles, outcome_codes.T,
        cmap=cmap, norm=norm, shading="nearest",
    )
    axes[0].set_title("Transition outcomes")
    axes[0].legend(
        handles=[
            Patch(facecolor=color, label=label)
            for _, label, color in categories
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        fontsize=8,
    )

    # Out-of-map returns still have a valid computed return velocity.
    returned = np.isin(outcomes, ["return", "out_of_map"])
    return_velocities = np.ma.masked_where(
        ~returned, table["next_velocities"]
    )
    if np.any(returned):
        image = axes[1].pcolormesh(
            velocities, angles, return_velocities.T,
            cmap="viridis", shading="nearest",
        )
        fig.colorbar(image, ax=axes[1], label=r"$\dot{\theta}_{k+1}$ (rad/s)")
    else:
        axes[1].text(
            0.5, 0.5, "No section returns",
            ha="center", va="center", transform=axes[1].transAxes,
        )
    axes[1].set_title("Velocity at the next section crossing")

    for ax in axes:
        ax.set(
            xlabel=r"Initial velocity $\dot{\theta}_k$ (rad/s)",
            ylabel=r"Angle of attack $\alpha$ (rad)",
            xlim=(velocities[0], velocities[-1]),
            ylim=(angles[0], angles[-1]),
        )
    return fig, axes


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


def plot_phase_portrait(params):
    """Plot passive stance dynamics and the forward Poincare section."""
    theta_values = np.linspace(-0.6, 0.6, 101)
    omega_values = np.linspace(-2.0, 2.0, 101)
    theta_grid, omega_grid = np.meshgrid(theta_values, omega_values)

    # Passive stance: ankle torque is zero.
    theta_dot = omega_grid
    omega_dot = (
        params["gravity"] / params["length"] * np.sin(theta_grid)
    )

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.streamplot(
        theta_values,
        omega_values,
        theta_dot,
        omega_dot,
        color="0.55",
        density=1.2,
        linewidth=0.8,
        arrowsize=1.0,
    )

    # Only the positive-velocity half-line is the forward section.
    ax.axvline(0.0, color="royalblue", linestyle=":", alpha=0.5)
    ax.plot(
        [0.0, 0.0],
        [0.0, omega_values[-1]],
        color="royalblue",
        linewidth=2.5,
        label=r"Poincaré section: $\theta=0,\ \dot{\theta}>0$",
    )

    alpha = params["angle_of_attack"]
    gamma = params["incline"]
    ax.axvline(
        gamma + alpha,
        color="darkorange",
        linestyle="--",
        label=r"Forward impact angle: $\theta=\gamma+\alpha$",
    )
    ax.axvline(
        gamma - alpha,
        color="purple",
        linestyle="--",
        label=r"Backward impact angle: $\theta=\gamma-\alpha$",
    )

    # Upright rest is an equilibrium, not a transverse section crossing.
    ax.plot(
        0.0, 0.0,
        marker="o",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=7,
        linestyle="none",
        zorder=5,
        label="Upright equilibrium",
    )

    ax.set(
        xlabel=r"$\theta$ (rad)",
        ylabel=r"$\dot{\theta}$ (rad/s)",
        title="Passive stance phase portrait",
        xlim=(theta_values[0], theta_values[-1]),
        ylim=(omega_values[0], omega_values[-1]),
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    return fig, ax


def draw_frame(index):
    # The massless swing leg is repositioned instantaneously at each impact.
    model.visualize(state_traj[:, index], params, ax=ax)
    ax.set_title(f"t = {time_traj[index]:.2f} s")


# RoA sweep: 50 * 50 = 2,500 initial conditions; zero is between grid points.
theta_values = np.linspace(-0.4, 0.4, 50)
omega_values = np.linspace(-1.5, 1.5, 50)
roa = estimate_roa(
    theta_values, omega_values, params, timestep=0.005, sim_time=5.0,
)
plot_roa(theta_values, omega_values, roa)
# Optional timestep/horizon rechecks are in assignment_2_validation.py.

validation_results = compare_policy_grids(
    params,
    (theta_values, omega_values, roa),
    build_transition_table,
    compute_minimum_step_policy,
    rollout_policy,
)

print("\nGrid | Matching step counts | Agreement")
for grid in sorted({row["grid"] for row in validation_results}):
    rows = [row for row in validation_results if row["grid"] == grid]
    matched = sum(row["matches"] for row in rows)
    print(
        f"{grid[0]} x {grid[1]} | "
        f"{matched}/{len(rows)} | "
        f"{100 * matched / len(rows):.2f}%"
    )

output = Path("output/assignment_2")
output.mkdir(parents=True, exist_ok=True)
validation_path = output / "grid_validation.csv"
with validation_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(validation_results[0]))
    writer.writeheader()
    writer.writerows(validation_results)
print(f"Saved {validation_path}")

# Passive transitions from the forward theta = 0 section.
velocity_values = np.linspace(
    0.0, np.sqrt(2 * params["gravity"] / params["length"]), 101,
)
alpha_values = np.linspace(np.pi / 8, np.pi / 7, 21)
transition_table = build_transition_table(
    velocity_values,
    alpha_values,
    params,
    roa_data=(theta_values, omega_values, roa),
)
table_fig, table_axes = plot_transition_table(transition_table)
policy = compute_minimum_step_policy(transition_table)
policy_fig, policy_axes = plot_minimum_step_policy(policy)
status, actual_steps, policy_history = rollout_policy(
    initial_velocity=3.0,
    policy=policy,
    params=params,
    roa_data=(theta_values, omega_values, roa),
)
print(f"Policy rollout: {status}, {actual_steps} footsteps")

outcomes, counts = np.unique(transition_table["outcomes"], return_counts=True)
for outcome, count in zip(outcomes, counts):
    print(f"{outcome}: {count}")

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
animation.save(output / "walker.gif", writer=PillowWriter(fps=fps))

# To save an MP4 instead, install FFmpeg and use:
# animation.save(output / "walker.mp4", writer="ffmpeg", fps=fps)
print(f"Saved {output / 'walker.gif'} ({completed_steps} footstrikes).")
phase_fig, phase_ax = plot_phase_portrait(params)
plt.show()
