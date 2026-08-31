import numpy as np
import matplotlib.pyplot as plt
import timeit

from models import pendulum as model
from integrators import rk4 as integrator

# Basic simulation of the pendulum

params = {
    "gravity": 9.81,  # gravity m/s^2)
    "length": 1,  # rod length (m)
    "mass": 0.2,  # point mass at end of rod (kg)
    "damping_coeff": 0.0,  # damping coefficient (kg*m^2/s)
}


# some set-up
initial_state = np.array([np.pi / 4, 0.0])

timestep = 4.3e-2
sim_time = 5.0

final_timestep = None
final_time_traj = None
final_kinetic_energy = None
final_potential_energy = None
final_total_energy = None

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state

# simulation loop
state_traj = integrator.rk4_int(model.dynamics, time_traj, state_traj, timestep, params)

# sanity check the energies: since there is no actuation, and no damping, total energy should stay
# constant. If we turn on the damping coefficient, it should slowly bleed out energy until it comes to
# a stand-still.

kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)
total_energy = kinetic_energy + potential_energy

# print(timeit.timeit())

plt.figure()
plt.plot(time_traj, potential_energy, label="Potential energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Pendulum energy")
plt.legend()
plt.tight_layout()

# TODO: make a phase portrait plot

plt.figure()
plt.plot(state_traj[0, :], state_traj[1, :])
plt.xlabel("Angle (rad)")
plt.ylabel("Angular velocity (rad/s)")
plt.title("Phase portrait")
plt.grid(True)
plt.tight_layout()
plt.show()