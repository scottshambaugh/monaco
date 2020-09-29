## [PyMonteCarlo](../../) - [Examples](../)

### Pandemic
This example simulates the spread of an epidemic through a social network. 
Specifically, it implements a SIR model onto a scale-free undirected network, 
in an attempt to replicate the "B+" scenario in the following paper:    
[Szabó, Gyula M. "Propagation and mitigation of epidemics in a scale-free network." arXiv preprint arXiv:2004.00067 (2020).](https://arxiv.org/abs/2004.00067)

The probability of infection along each edge is randomized in the range 
[0.28, 0.32], as is the graph itself, the starting infected nodes, and the odds 
of infection ocurring along an edge at every timestep. 

Read more about the Barabási–Albert scale free network model [on Wikipedia](https://en.wikipedia.org/wiki/Barabási–Albert_model).

<p float="left" align="center">
<img width="440" height="300" src="./network_graph.png">  
<img width="440" height="300" src="./cum_infections_vs_time.png">
</p>
