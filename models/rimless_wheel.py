import numpy as np


def pendulum_dynamics(t, state, params):
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    angle = state[0]
    angular_velocity = state[1]

    angular_acceleration = (
        mass * gravity * length * np.sin(angle)
    ) / (mass * length**2)

    state_derivative = np.array([angular_velocity, angular_acceleration])
    return state_derivative


def detect_event(previous_state, state, params):
    event_angle = np.pi / params["N"] + params["Incline"]
    return (
        previous_state[0] < event_angle
        and state[0] >= event_angle
        and state[1] > 0
    )


def reset_impact(state, params):
    alpha = np.pi / params["N"]
    post_impact_angle = alpha - params["Incline"]
    reset_state = np.array(state, copy=True)
    reset_state[0] = -post_impact_angle
    reset_state[1] = np.cos(2 * alpha) * state[1]
    return reset_state


def calculate_energy(state, height_traj, params):
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    angle = state[0]
    angular_velocity = state[1]

    kinetic_energy = 0.5 * mass * (length * angular_velocity) ** 2
    potential_energy = mass * gravity * (length * np.cos(angle) + height_traj)
    return kinetic_energy, potential_energy


def walking_converged(impact_velocities, window=5, tol=1e-3):
    if len(impact_velocities) < window:
        return False

    recent = np.asarray(impact_velocities[-window:])
    return np.max(np.abs(np.diff(recent))) < tol


def refine_impact(previous_state, t, coarse_dt, fine_dt, params, integrator):
    state = np.array(previous_state, copy=True)
    local_t = t

    n_fine_steps = int(np.ceil(coarse_dt / fine_dt))

    for _ in range(n_fine_steps):
        next_state = integrator(pendulum_dynamics, local_t, state, fine_dt, params)

        if detect_event(state, next_state, params):
            return next_state

        state = next_state
        local_t += fine_dt

    return state


def next_impact_velocity(
    current_velocity,
    params,
    timestep,
    impact_timestep,
    integrator,
    post_impact_angle,
    pendulum_dynamics_func,
    detect_event_func,
    refine_impact_func,
    reset_impact_func,
):
    state = np.array([-(post_impact_angle), current_velocity], dtype=float)
    t = 0.0

    while True:
        next_state = integrator(pendulum_dynamics_func, t, state, timestep, params)

        if detect_event_func(state, next_state, params):
            refined_state = refine_impact_func(
                state,
                t,
                timestep,
                impact_timestep,
                params,
                integrator,
            )
            reset_state = reset_impact_func(refined_state, params)
            return float(reset_state[1])

        state = next_state
        t += timestep

def estimate_floquet(fixed_point, epsilon, params, timestep, impact_timestep, integrator, post_impact_angle):
    P_minus = next_impact_velocity(
        fixed_point - epsilon,
        params,
        timestep,
        impact_timestep,
        integrator,
        post_impact_angle,
        pendulum_dynamics,
        detect_event,
        refine_impact,
        reset_impact,
    )
    P_plus = next_impact_velocity(
        fixed_point + epsilon,
        params,
        timestep,
        impact_timestep,
        integrator,
        post_impact_angle,
        pendulum_dynamics,
        detect_event,
        refine_impact,
        reset_impact,
    )
    return (P_plus - P_minus) / (2 * epsilon)