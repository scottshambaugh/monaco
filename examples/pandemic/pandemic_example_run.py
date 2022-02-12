import numpy as np
import networkx as nx

def pandemic_example_run(nnodes, m0, p, nsteps, n_infected_init, open_scenario=True, seed=12059257):
    # nnodes: total number of nodes
    # m0: max number of connections made by each new node
    # p: the probability a node becomes infected if it shares an edge with an
    #    infected node
    # n_infected_init: number of nodes infected at the first timestep
    # open_scenario: if True, a random node will be attempt to be infected every
    #                timestep with probability p

    nodes_status = np.zeros((nsteps+1, nnodes))  # S = 0, I = 1, R = 2
    social_graph = nx.barabasi_albert_graph(nnodes, m0, seed=seed)
    infection_dict = dict()
    generator = np.random.RandomState(seed)

    # Initialize initial infections
    initial_infected_nodes = generator.randint(0, nnodes-1, n_infected_init)
    for node in initial_infected_nodes:
        infect_node(nodes_status, 0, node)

    # Run the epidemic spread
    for t in range(1, nsteps+1):

        if open_scenario:
            generator.randint(0, nnodes-1, n_infected_init)
            if generator.rand(1) < p:
                infect_node(nodes_status, t, node)

        for node in range(nnodes):
            if nodes_status[t-1, node] == 1:
                neighbors = np.array(list(social_graph.neighbors(node)))
                infections = generator.rand(len(neighbors)) < p
                infection_dict[node] \
                    = neighbors[np.logical_and(nodes_status[t, neighbors] == 0, infections)]
                for neighbor in infection_dict[node]:
                    infect_node(nodes_status, t, neighbor)

    # Build the infection graph
    infection_graph = nx.DiGraph(infection_dict)

    return (social_graph, nodes_status, infection_graph)


def infect_node(nodes_status, t, node):
    if nodes_status[t, node] == 0:  # S
        nodes_status[t, node] = 1  # I
        if t < len(nodes_status):
            nodes_status[t+1:, node] = 2  # R


'''
### Test ###
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    nnodes = 50000
    m0 = 2
    p = 0.30
    nsteps = 30
    n_infected_init = 3
    open_scenario = True
    seed = 129251

    (social_graph, nodes_status, infection_graph) \
        = pandemic_example_run(nnodes=nnodes, m0=m0, p=p, nsteps=nsteps,
                               n_infected_init=n_infected_init,
                               open_scenario=open_scenario, seed=seed)

    # # Plot social graph (diabled by default since it freezes for large nnodes)
    # from matplotlib import colors
    # cmap = colors.ListedColormap(['silver', 'red', 'black'])
    # pos = nx.spring_layout(social_graph)
    # fig = plt.figure()
    # nodes_status_final = np.array(nodes_status[-1,:])
    # nx.draw(social_graph, node_color=nodes_status_final, cmap=cmap, pos=pos, vmin=0, vmax=2)
    # nx.draw(infection_graph, node_color=nodes_status_final[list(infection_graph.nodes)],
    #         cmap=cmap, pos=pos, vmin=0, vmax=2)
    # plt.savefig('network_graph.png')
    # plt.show()

    # Show distribution of infections vs degree (should be linear in log-log)
    superspreader_degree = []
    nodes = np.array(list(range(nnodes)))
    for t in range(nsteps+1):
        infected = nodes[[i for i, x in enumerate(nodes_status[t,:]) if x == 1]]
        infected_degree = list(dict(social_graph.degree(infected)).values())
        if infected_degree == []:
            infected_degree = [0,]
        superspreader_degree.append(max(infected_degree))
    fig = plt.figure()
    plt.scatter(range(nsteps+1), superspreader_degree, color='k')
    plt.xlabel('Timestep')
    plt.ylabel('Degree of Biggest Spreader')

    # Show distribution of node degree (should be linear in log-log)
    from collections import Counter
    degrees = Counter(sorted(dict(social_graph.degree()).values()))
    fig = plt.figure()
    plt.scatter(degrees.keys(), degrees.values(), color='k')
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.ylabel('Number of Nodes')
    plt.xlabel('Degree of Node')
    plt.savefig('scale_free_connectivity.png')

    # Show distribution of node degree (should be linear in log-log)
    from collections import Counter
    outdegrees = Counter(sorted(dict(infection_graph.out_degree()).values()))
    fig = plt.figure()
    plt.scatter(outdegrees.keys(), outdegrees.values(), color='k')
    plt.gca().set_yscale('log')
    plt.ylabel('Number of Infected Nodes')
    plt.xlabel('Infections from Node')

    # Plot SIR Curve
    nS = np.count_nonzero(nodes_status == 0, axis=1)
    nI = np.count_nonzero(nodes_status == 1, axis=1)
    nR = np.count_nonzero(nodes_status == 2, axis=1)
    fig = plt.figure()
    plt.plot(range(nsteps+1), nS, color='silver')
    plt.plot(range(nsteps+1), nI, color='C1')
    plt.plot(range(nsteps+1), nR, color='black')
    plt.legend(['S','I','R'])
    plt.xlabel('Timestep')
    plt.ylabel('n')
    plt.savefig('sir_curve.png')
#'''
