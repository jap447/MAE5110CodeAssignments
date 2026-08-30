import numpy as np


def rk4_int(dynamics, time_traj, state_traj, dt, params):
    for step, t in enumerate(time_traj[:-1]):
        k1 = dt * dynamics(t, state_traj[:, step], params)
        k2 = dt * dynamics(t + dt / 2, state_traj[:, step] + k1 / 2, params)
        k3 = dt * dynamics(t + dt / 2, state_traj[:, step] + k2 / 2, params)
        k4 = dt * dynamics(t + dt, state_traj[:, step] + k3, params)
        state_traj[:, step + 1] = state_traj[:, step] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return state_traj