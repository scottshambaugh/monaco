import numpy as np


def pandemic_example_postprocess(case, social_graph, nodes_status, infection_graph):
    nnodes = case.siminput[0]
    nsteps = case.siminput[3]
    case.addOutVal("Timestep", list(range(nsteps + 1)))

    nS = np.count_nonzero(nodes_status == 0, axis=1)
    nI = np.count_nonzero(nodes_status == 1, axis=1)
    nR = np.count_nonzero(nodes_status == 2, axis=1)

    case.addOutVal("Number Susceptible", nS)
    case.addOutVal("Number Infected", nI)
    case.addOutVal("Number Recovered", nR)

    case.addOutVal("Cumulative Infections", nR + nI)
    case.addOutVal("Proportion Infected", (nR + nI) / nnodes)

    herd_immunity_threshold = (nR[-1] + nI[-1]) / nnodes
    case.addOutVal("Herd Immunity Threshold", herd_immunity_threshold)

    superspreader_degree = []
    nodes = np.array(list(range(nnodes)))
    for t in range(nsteps + 1):
        infected = nodes[[i for i, x in enumerate(nodes_status[t, :]) if x == 1]]
        infected_degree = list(dict(social_graph.degree(infected)).values())
        if infected_degree == []:
            infected_degree = [
                0,
            ]
        superspreader_degree.append(max(infected_degree))
    case.addOutVal("Superspreader Degree", superspreader_degree)
    case.addOutVal("Max Superspreader Degree", max(superspreader_degree))
