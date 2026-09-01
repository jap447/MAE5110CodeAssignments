import numpy as np


def rk4_int(dynamics, time_traj, state_traj, dt, params):
    for step, t in enumerate(time_traj[:-1]):
        k1 = dt * dynamics(t, state_traj[:, step], params)
        k2 = dt * dynamics(t + dt / 2, state_traj[:, step] + k1 / 2, params)
        k3 = dt * dynamics(t + dt / 2, state_traj[:, step] + k2 / 2, params)
        k4 = dt * dynamics(t + dt, state_traj[:, step] + k3, params)
        state_traj[:, step + 1] = state_traj[:, step] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return state_traj

def rk4_single(dynamics, time_traj, state_traj, dt, params, event_handler=None):
    for step, t in enumerate(time_traj[:-1]):
        k1 = dt * dynamics(t, state_traj[:, step], params)
        k2 = dt * dynamics(t + dt / 2, state_traj[:, step] + k1 / 2, params)
        k3 = dt * dynamics(t + dt / 2, state_traj[:, step] + k2 / 2, params)
        k4 = dt * dynamics(t + dt, state_traj[:, step] + k3, params)
        state_traj[:, step + 1] = state_traj[:, step] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
        # Handle bounces after the step completes
        if state_traj[0, step + 1] <= 0 and state_traj[1, step + 1] < 0:
            state_traj[1, step + 1] = -params["restitution_coeff"] * state_traj[1, step + 1]
            state_traj[0, step + 1] = 0

    return state_traj