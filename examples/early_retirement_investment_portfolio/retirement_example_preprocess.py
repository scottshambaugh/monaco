def retirement_example_preprocess(case):
    nyears = case.constvals['nyears']
    yearly_returns = []
    for i in range(nyears):
        yearly_returns.append(case.invals[f'Year {i} Returns'].val)

    beginning_investments = case.invals['Beginning Balance'].val
    yearly_spending = 50000  # total number of nodes

    return (yearly_returns, beginning_investments, yearly_spending)
