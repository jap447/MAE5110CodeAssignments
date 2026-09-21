from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from integrators import rk4 as integrator
from models import inverted_pendulum_walker as model
from integrators import rk4 as integrator

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
    "Kp": 100.0,  # N m/rad
    "Kd": 10.0,  # N m/(rad/s)
}

initial_state = np.array([0.0, 3.0])
timestep = 1e-4
sim_time = 3.0
desired_number_of_steps = 3

n_timesteps = round(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state
completed_steps = 0

# Simulation loop. Replace this Euler step with your own integrator as needed.
def simulate(
    initial_state,
    params,
    timestep,
    sim_time,
    balance=False,
    *,
    impact_timestep=1e-5,
    desired_number_of_steps=None,
):

    params = params.copy()
    time_traj = np.append(np.arange(0.0, sim_time, timestep), sim_time)
    state_traj = np.zeros((2, len(time_traj)))
    state_traj[:, 0] = initial_state
    completed_steps = 0

    for step, t in enumerate(time_traj[:-1]):
        current_state = state_traj[:, step]
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


def draw_frame(index):
    # The massless swing leg is repositioned instantaneously at each impact.
    model.visualize(state_traj[:, index], params, ax=ax)
    ax.set_title(f"t = {time_traj[index]:.2f} s")


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
