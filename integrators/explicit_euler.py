import numpy as np


def euler_int(dynamics, time_traj, state_traj, dt, params):
    for step, t in enumerate(time_traj[:-1]):
        state_traj[:, step + 1] = state_traj[:, step] + dt * dynamics(
            t, state_traj[:, step], params
        )

    return state_traj