import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from integrators import rk4 as integrator
from models import rimless_wheel as model


# Sweep configuration (coarse defaults; increase resolution if desired)
N_values = list(range(6, 13))
incline_values = np.linspace(np.pi/48, np.pi/8, 8)

# Simulation timing
# Use a coarse step for the RoA sweep to keep runtime manageable.
timestep = 1e-2
impact_timestep = 1e-5
sim_time = 10.0
n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep

# Output folder
out_dir = os.path.join(os.path.dirname(__file__), "../sweep_output")
os.makedirs(out_dir, exist_ok=True)


# RoA colormap (same classifications as assignment_1.py)
ROA_CMAP = ListedColormap([
    "orange",
    "royalblue",
    "red",
    "purple",
    "gray",
])
ROA_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ROA_CMAP.N)


def save_roa_map(theta_grid, omega_grid, roa_grid, label):
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
    cbar.ax.set_yticklabels([
        "failed before first impact",
        "one step then failed",
        "multiple steps then failed",
        "walking limit cycle",
        "unresolved",
    ])
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$\dot{\theta}$ (rad/s)")
    ax.set_title(label)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{label.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}.png"), dpi=200)
    plt.close(fig)


def sweep_incline(params):
    floquet_vals = []
    saved_maps = []

    N_orig = params.get("N", 8)

    for inc in incline_values:
        params["Incline"] = float(inc)
        alpha_local = np.pi / params["N"]
        theta_vals = np.linspace(-(alpha_local - params["Incline"]), alpha_local + params["Incline"], 20)
        omega_vals = np.linspace(0.0, 2.5, 20)
        theta_grid_local, omega_grid_local = np.meshgrid(theta_vals, omega_vals)

        roa_grid = model.compute_roa(
            theta_grid_local,
            omega_grid_local,
            n_timesteps,
            time_traj,
            timestep,
            impact_timestep,
            params,
            integrator,
            model,
        )

        # save actual RoA heatmap
        save_roa_map(theta_grid_local, omega_grid_local, roa_grid, f"RoA for incline = {inc:.4f} rad")
        saved_maps.append((inc, theta_grid_local, omega_grid_local, roa_grid))

        # find fixed point by iterating impacts
        post_angle = alpha_local - params["Incline"]
        v = 1.0
        for _ in range(300):
            v_next = model.next_impact_velocity(
                v,
                params,
                timestep,
                impact_timestep,
                integrator,
                post_angle,
                model.pendulum_dynamics,
                model.detect_event,
                model.refine_impact,
                model.reset_impact,
            )
            if abs(v_next - v) < 1e-8:
                break
            v = v_next

        floq = model.estimate_floquet(v, 1e-4, params, timestep, impact_timestep, integrator, post_angle)
        floquet_vals.append(float(floq))

    params["N"] = N_orig
    return np.asarray(incline_values), np.asarray(floquet_vals), saved_maps


def sweep_spokes(params):
    floquet_vals = []
    saved_maps = []

    N_orig = params.get("N", 8)
    inc_orig = params.get("Incline", 0.0)

    for Nval in N_values:
        params["N"] = int(Nval)
        params["Incline"] = inc_orig
        alpha_local = np.pi / params["N"]
        theta_vals = np.linspace(-(alpha_local - params["Incline"]), alpha_local + params["Incline"], 25)
        omega_vals = np.linspace(0.0, 2.5, 25)
        theta_grid_local, omega_grid_local = np.meshgrid(theta_vals, omega_vals)

        roa_grid = model.compute_roa(
            theta_grid_local,
            omega_grid_local,
            n_timesteps,
            time_traj,
            timestep,
            impact_timestep,
            params,
            integrator,
            model,
        )

        save_roa_map(theta_grid_local, omega_grid_local, roa_grid, f"RoA for N = {Nval}")
        saved_maps.append((Nval, theta_grid_local, omega_grid_local, roa_grid))

        post_angle = alpha_local - params["Incline"]
        v = 1.0
        for _ in range(300):
            v_next = model.next_impact_velocity(
                v,
                params,
                timestep,
                impact_timestep,
                integrator,
                post_angle,
                model.pendulum_dynamics,
                model.detect_event,
                model.refine_impact,
                model.reset_impact,
            )
            if abs(v_next - v) < 1e-8:
                break
            v = v_next

        floq = model.estimate_floquet(v, 1e-4, params, timestep, impact_timestep, integrator, post_angle)
        floquet_vals.append(float(floq))

    params["N"] = N_orig
    params["Incline"] = inc_orig
    return np.asarray(N_values), np.asarray(floquet_vals), saved_maps


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
