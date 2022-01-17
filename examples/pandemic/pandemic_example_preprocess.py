def pandemic_example_preprocess(case):
    nnodes = 1000         # total number of nodes
    m0 = 2                # m0: max number of connections made by each new node
    # p: the probability a node becomes infected if it shares an edge with an infected node
    p = case.invals['Probability of Infection'].val,
    nsteps = 30           # number of timesteps to run the simulation for
    n_infected_init = 3   # n_infected_init: number of nodes infected at the first timestep
    # open_scenario: if True, a random node will attempt to be infected every timestep with
    #                probability p
    open_scenario = True
    seed = case.seed    # seed: random seed for the run

    return (nnodes, m0, p, nsteps, n_infected_init, open_scenario, seed)
