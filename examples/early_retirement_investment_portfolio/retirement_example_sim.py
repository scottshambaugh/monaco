import numpy as np
import pandas as pd

def retirement_example_sim(yearly_returns, beginning_investments, yearly_spending):
    # nnodes: total number of nodes
    # m0: max number of connections made by each new node
    # p: the probability a node becomes infected if it shares an edge with an infected node
    # n_infected_init: number of nodes infected at the first timestep
    # open_scenario: if True, a random node will be attempt to be infected every timestep with probability p
    
    nyears = len(yearly_returns)
    
    starting_balance = np.ones(nyears)*beginning_investments
    ending_balance = np.zeros(nyears)
    
    for i in range(nyears):
        if i > 0:
            starting_balance[i] = ending_balance[i-1]
        ending_balance[i] = max(starting_balance[i]*(1+yearly_returns[i]) - yearly_spending, 0)

    dates = pd.date_range(start='2020-01-01', periods=nyears, freq='YS')
    df = pd.DataFrame({'Returns'         : yearly_returns,
                       'Spending'        : yearly_spending,
                       'Starting Balance': starting_balance,
                       'Ending Balance'  : ending_balance},\
                      index = dates)

    return (df)


'''
### Test ###
yearly_returns = np.ones(50)*0.07
beginning_investments = 1000000
yearly_spending = 50000
df = retirement_example_sim(yearly_returns, beginning_investments, yearly_spending)
print(df.head)
import matplotlib.pyplot as plt
plt.plot(df.index.values, df['Ending Balance'].values)
#'''
