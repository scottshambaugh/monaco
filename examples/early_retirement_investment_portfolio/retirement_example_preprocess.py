def retirement_example_preprocess(mccase):
    nyears = mccase.constvals['nyears']
    yearly_returns = []
    for i in range(nyears):
        yearly_returns.append(mccase.mcinvals[f'Year {i} Returns'].val)

    beginning_investments = mccase.mcinvals['Beginning Balance'].val
    yearly_spending = 50000  # total number of nodes

    return (yearly_returns, beginning_investments, yearly_spending)
