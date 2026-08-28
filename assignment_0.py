import numpy as np
import matplotlib.pyplot as plt

from models import pendulum as model

# Basic simulation of the pendulum

params = {
    "gravity": 9.81,  # gravity m/s^2)
    "length": 1,  # rod length (m)
    "mass": 0.2,  # point mass at end of rod (kg)
    "damping_coeff": 0.0,  # damping coefficient (kg*m^2/s)
}


# some set-up
initial_state = np.array([np.pi / 4, 0.0])

timesteps = np.logspace(-5, -2, 20)
sim_time = 5.0

max_error = 1e-3

energy_errors = []

final_timestep = None
final_time_traj = None
final_kinetic_energy = None
final_potential_energy = None
final_total_energy = None


for timestep in timesteps:

    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state

    # simulation loop
    for step, t in enumerate(time_traj[:-1]):
        state_traj[:, step + 1] = state_traj[:, step] + timestep * model.dynamics(
            t, state_traj[:, step], params
        )

    # sanity check the energies: since there is no actuation, and no damping, total energy should stay
    # constant. If we turn on the damping coefficient, it should slowly bleed out energy until it comes to
    # a stand-still.

    kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)
    total_energy = kinetic_energy + potential_energy

    energy_error = np.max(np.abs(total_energy - total_energy[0]))

    energy_errors.append(energy_error)

    print(f"dt = {timestep:.6f} s, max energy error = {energy_error:.6e} J")

    if energy_error > max_error:
        print("\nEnergy error threshold exceeded.")
        print(f"First failing timestep: {timestep:.6f} s")

        break

    final_timestep = timestep
    final_time_traj = time_traj
    final_kinetic_energy = kinetic_energy
    final_potential_energy = potential_energy
    final_total_energy = total_energy

print(f"Largest acceptable timestep: "f"{final_timestep:.6f} s")

plt.figure()
plt.plot(final_time_traj, final_potential_energy, label="Potential energy")
plt.plot(final_time_traj, final_kinetic_energy, label="Kinetic energy")
plt.plot(final_time_traj, final_potential_energy + final_kinetic_energy, label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Pendulum energy")
plt.legend()
plt.tight_layout()

tested_timesteps = timesteps[:len(energy_errors)]

plt.figure()
plt.plot(tested_timesteps, energy_errors, "o-")
plt.axhline(max_error, linestyle="--", label="Error threshold")
plt.xlabel("Timestep (s)")
plt.ylabel("Maximum energy error (J)")
plt.title("Energy Error vs. Timestep")
plt.grid(True)
plt.legend()
plt.tight_layout()

# TODO: make a phase portrait plot
