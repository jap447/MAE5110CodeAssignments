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


def compare_policy_grids(
    params, roa_data, build_transition_table, compute_minimum_step_policy,
    rollout_policy, grid_sizes=((51, 11), (101, 21), (201, 41)),
    test_velocities=None,
):
    """Compare predictions with finer-timestep rollouts on shared test states.

    Pass the three functions from assignment_2.py, as with recheck_roa.
    Returns per-velocity records for comparison with the finest policy.
    No-policy results are not counted as correctly predicted failures.
    """
    if test_velocities is None:
        test_velocities = np.arange(0.25, 4.26, 0.25)
    validation_results = []
    for n_velocities, n_angles in grid_sizes:
        velocity_grid = np.linspace(
            0.0, np.sqrt(2 * params["gravity"] / params["length"]), n_velocities,
        )
        angle_grid = np.linspace(np.pi / 8, np.pi / 7, n_angles)
        table = build_transition_table(
            velocity_grid, angle_grid, params, roa_data,
        )
        grid_policy = compute_minimum_step_policy(table)
        predicted_recoverable = 0
        successful_rollouts = 0
        matching_counts = 0
        for velocity in test_velocities:
            i = np.argmin(np.abs(velocity_grid - velocity))
            predicted_steps = grid_policy["steps"][i]
            status, actual_steps, _ = rollout_policy(
                velocity, grid_policy, params, roa_data, timestep=0.0005,
            )
            predicted_success = np.isfinite(predicted_steps)
            actual_success = status == "success"
            matches = (
                predicted_success and actual_success
                and actual_steps == predicted_steps
            )
            predicted_recoverable += int(predicted_success)
            successful_rollouts += int(actual_success)
            matching_counts += int(matches)
            validation_results.append({
                "grid": (n_velocities, n_angles),
                "initial_velocity": velocity,
                "predicted_steps": predicted_steps,
                "status": status,
                "actual_steps": actual_steps,
                "matches": bool(matches),
            })
        print(
            f"{n_velocities} x {n_angles}: "
            f"{predicted_recoverable}/{len(test_velocities)} predicted recoverable, "
            f"{successful_rollouts}/{len(test_velocities)} reached RoA, "
            f"{matching_counts}/{len(test_velocities)} matched step counts"
        )
    return validation_results
