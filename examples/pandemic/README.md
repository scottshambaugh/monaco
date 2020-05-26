## [PyMonteCarlo](../../) - [Examples](../)

### Pandemic
This example simulates the spread of an epidemic through a social network. 
Specifically, it implements a SIR model onto a scale-free undirected network, 
in an attempt to replicate the "B+" scenario in the following paper:    
[Szabó, Gyula M. "Propagation and mitigation of epidemics in a scale-free network." arXiv preprint arXiv:2004.00067 (2020).](https://arxiv.org/abs/2004.00067)

The inputs to the simulation are not randomized, but the graph itself, the 
starting infected nodes, and the odds of infection ocurring along an edge at 
every timestep are. 