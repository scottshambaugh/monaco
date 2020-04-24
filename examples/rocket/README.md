## [PyMonteCarlo](../../) - [Examples](../)

### Rocket
This example simulates the flight of a model rocket, as an example of 
monte-carlo applied to physics or engineering problems.

The simulation only models 3-degrees of freedom and does not perform robust 
integration. But it does model aerodynamic and wind effects, varying thrust and 
mass loss curves, and the deployment of a parachute shortly after apogee, which 
has a 10% chance of failure. These produce enough nonlinear dynamics to make 
for interesting results. 