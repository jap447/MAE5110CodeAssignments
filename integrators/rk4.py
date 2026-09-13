def rk4(dynamics, time, state, timestep, params):
    """Advance one state by one fixed fourth-order Runge-Kutta step."""
    k1= dynamics(time, state, params)
    k2 = dynamics(time + timestep / 2, state + timestep * k1 / 2,
        params,
    )
    k3 = dynamics(time + timestep / 2, state + timestep * k2 / 2,
        params,
    )
    k4 = dynamics(time + timestep,state + timestep * k3,
        params,
    )
    rk4state = state + (timestep / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return rk4state