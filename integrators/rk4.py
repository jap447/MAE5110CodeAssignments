def rk4(dynamics, time, state, timestep, params):
    """Advance one state by one fixed fourth-order Runge-Kutta step."""
    first_slope = dynamics(time, state, params)
    second_slope = dynamics(
        time + timestep / 2,
        state + timestep * first_slope / 2,
        params,
    )
    third_slope = dynamics(
        time + timestep / 2,
        state + timestep * second_slope / 2,
        params,
    )
    fourth_slope = dynamics(
        time + timestep,
        state + timestep * third_slope,
        params,
    )

    return state + timestep * (
        first_slope + 2 * second_slope + 2 * third_slope + fourth_slope
    ) / 6