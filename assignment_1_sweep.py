import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from integrators import rk4 as integrator
from models import rimless_wheel as model

# Sweep configuration (coarse defaults; increase resolution if desired)
N_values = list(range(6, 13))
incline_values = np.linspace(np.pi / 48, np.pi / 8, 8)

# Simulation timing
# Use a coarse step for the RoA sweep to keep runtime manageable.
timestep = 1e-2
impact_timestep = 1e-5
sim_time = 10.0
n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep

# Output folder
out_dir = os.path.join(os.path.dirname(__file__), "../sweep_output")


# RoA colormap (same classifications as assignment_1.py)
ROA_CMAP = ListedColormap(
    [
        "orange",
        "royalblue",
        "red",
        "purple",
        "gray",
    ]
)
ROA_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ROA_CMAP.N)


def save_roa_map(theta_grid, omega_grid, roa_grid, label):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.pcolormesh(
        theta_grid,
        omega_grid,
        roa_grid,
        cmap=ROA_CMAP,
        norm=ROA_NORM,
        shading="auto",
    )
    cbar = fig.colorbar(image, ax=ax, ticks=[0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels(
        [
            "failed before first impact",
            "one step then failed",
            "multiple steps then failed",
            "walking limit cycle",
            "unresolved",
        ]
    )
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$\dot{\theta}$ (rad/s)")
    ax.set_title(label)
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            out_dir,
            f"{label.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}.png",
        ),
        dpi=200,
    )
    plt.close(fig)


def sweep_parameter(params, parameter, values, grid_size):
    """Sweep independent parameter copies and return values, RoA, and Floquet data."""
    floquet_values = []
    roa_fractions = []
    saved_maps = []

    for value in values:
        sweep_params = {**params, parameter: value}
        alpha = np.pi / sweep_params["N"]
        incline = sweep_params["Incline"]
        post_angle = alpha - incline
        theta_values = np.linspace(-post_angle, alpha + incline, grid_size)
        omega_values = np.linspace(0.0, 2.5, grid_size)
        theta_grid, omega_grid = np.meshgrid(theta_values, omega_values)

        roa_grid = model.compute_roa(
            theta_grid,
            omega_grid,
            n_timesteps,
            time_traj,
            timestep,
            impact_timestep,
            sweep_params,
            integrator,
            model,
        )
        label = (
            f"RoA for incline = {value:.4f} rad"
            if parameter == "Incline"
            else f"RoA for N = {value}"
        )
        save_roa_map(theta_grid, omega_grid, roa_grid, label)
        saved_maps.append((value, theta_grid, omega_grid, roa_grid))
        roa_fractions.append(np.mean(roa_grid == 3))

        velocity = 1.0
        for _ in range(300):
            next_velocity = model.next_impact_velocity(
                velocity,
                sweep_params,
                timestep,
                impact_timestep,
                integrator,
                post_angle,
            )
            converged = abs(next_velocity - velocity) < 1e-8
            velocity = next_velocity
            if converged:
                break

        floquet = model.estimate_floquet(
            velocity,
            1e-4,
            sweep_params,
            timestep,
            impact_timestep,
            integrator,
            post_angle,
        )
        floquet_values.append(float(floquet))

    return (
        np.asarray(values),
        np.asarray(roa_fractions),
        np.asarray(floquet_values),
        saved_maps,
    )


def sweep_incline(params):
    return sweep_parameter(params, "Incline", incline_values, grid_size=20)


def sweep_spokes(params):
    return sweep_parameter(params, "N", N_values, grid_size=25)


if __name__ == "__main__":
    params = {
        "gravity": 9.81,
        "length": 1.0,
        "mass": 0.2,
        "restitution_coeff": 0.0,
        "N": 8,
        "Incline": np.pi / 24,
    }

    inclines, roa_inc, floq_inc, incline_maps = sweep_incline(params)
    np.savetxt(
        os.path.join(out_dir, "sweep_incline_roa_floquet.csv"),
        np.vstack([inclines, roa_inc, floq_inc]).T,
        delimiter=",",
        header="incline,roa_fraction,floquet",
        comments="",
    )

    Ns, roa_N, floq_N, N_maps = sweep_spokes(params)
    np.savetxt(
        os.path.join(out_dir, "sweep_spokes_roa_floquet.csv"),
        np.vstack([Ns, roa_N, floq_N]).T,
        delimiter=",",
        header="N,roa_fraction,floquet",
        comments="",
    )

    # summarize sweeps
    plt.figure()
    plt.plot(inclines, roa_inc, "o-", label="RoA fraction")
    plt.plot(inclines, floq_inc, "s-", label="Floquet")
    plt.xlabel("Incline (rad)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "incline_sweep_summary.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(Ns, roa_N, "o-", label="RoA fraction")
    plt.plot(Ns, floq_N, "s-", label="Floquet")
    plt.xlabel("Number of spokes (N)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "spokes_sweep_summary.png"), dpi=200)
    plt.close()

    print(f"Saved sweep data in: {out_dir}")
    print("Inclines:", inclines)
    print("RoA fractions (incline):", roa_inc)
    print("Floquet values (incline):", floq_inc)
    print("N values:", Ns)
    print("RoA fractions (N):", roa_N)
    print("Floquet values (N):", floq_N)
