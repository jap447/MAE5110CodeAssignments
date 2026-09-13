# Assignment 1

## Rimless wheel model sanity checks:
![alt text](image-2.png)
In order to check the dynamic model I developed for the rimless wheel is correct, I decided to look at the energy behavior of the system. Plotting the kinetic, potential, and total energies of the system, I would expect to see a pattern in which the total energy starts decreasing at a steady rate as the systen reaches an equilibrium where the energy introduced by the downward slope of the ramp is equalized by the energy loss during the reset step, where the velocity of the center mass follows $\dot{\theta}^+ = \dot{\theta}^- \cos(2 \alpha)$. Due to the way in which the dynamics were setup to only check whether the angle of the spoke reaches its contact point, a case in which the mass cannot make it past the vertical and starts oscillating back and forward will look like a regular pendulum swinging back and forward, which is one of the failure cases considered in the following sections. Furthermore, for the stable walking case, I would expect the kinetic energy plot to show as a repeating pattern that does not decay with time, while the potential energy of the system drops with time (also with a repeating pattern).

## State  Space plot showing the region of attraction
![alt text](image.png)
For the region of attraction, I would expect three main stable long-term behaviors. The first would be stable walking as mentioned above. The second would be taking one step and stopping due to the system not having enough energy to complete the next step, and the third would be the inverted pendulum not reaching the vertical and falling back. From the image above, we can see the majority of the region of attractiopn is spanned by the stable walking case in purple. There are two other regions, a yellow section where the platform angle and initial velocity conbination is such that the spokeless wheel does not reach the first step, and a blue region where it stops after reaching the first step since the energy drop after impact is too high for the given slope and initial velocity, and it cannot reach thevertical during the next step and falls back. 

## Poincare map and 
![alt text](image-1.png)
The poincare map above represents how the angular velocities of subsequent crossings tend toward a fized point. It is possible to see how early in the simulation, the points are much further apart, and then start converging ultil intersecting the identity line at a fixed point of approximately 1.400013 rad/s. This corresponds to the point where the energy acquired by the change in height of the new fulcrum perfectly equalizes the energy loss due to the reset as mentioned in the first section. Furthermore, the floquet multiplier can be estimated by analyzing how different perturbances behave around this point. The floquet multiplier can be interpreted as the rate at which the convergence above happens, and it can be studied by looking at a couple different perturbations as seen below. 

![alt text](image-3.png)
This plit shows how the floquet multiplier estimates converge around a 1e-3 disturbance, which we can then use in order to estimate the value of the multiplier at 0.499933. This value is lower than 1 as expected given the convergence.

## Slope and spoke number sweep


