import numpy as np


def recheck_roa(
    theta_values, omega_values, roa, params, simulate, classify_balance,
    timestep=0.0025, sim_time=10.0,
):
    """Return the rechecked map and a mask of the points tested."""
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
            timestep=timestep,
            sim_time=sim_time,
            balance=True,
        )
        checked_roa[i, j] = classify_balance(time_check, state_check, params)

    return checked_roa, check_points
